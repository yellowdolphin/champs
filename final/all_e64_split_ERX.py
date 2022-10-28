# Hyperparameters, settings
learning_rate = 1e-4   # after conv to ~ 0.0025 (round 2), lower from 2e-4 to 0.8e-4
epochs = 10
chk_period = 10
val_period = 5
batch_size = 32

# Model parameter
max_node = 29
n_atom_types = 5
n_edge_features = 42
state_dim = 64
reg_dim = 256
T = 4
datadir = '../data/champs-final-inputs/'
picklefile = datadir + "all_inputs.p"
transfer_mp = False  # train new Readout


import os
import sys
import numpy as np
import random
import gzip
import pickle
from time import time
import pandas as pd
from tensorflow import set_random_seed
import tensorflow as tf

import keras
from keras.layers import (Input, Embedding, Dense, BatchNormalization,
                          Concatenate, Multiply, Add, Lambda)
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from nfp.layers import Squeeze, GatherAtomToBond, ReduceBondToAtom, ReduceAtomToMol
from nfp.models import GraphModel

from champs_util import *


# Get job name and cycle from shell.
name = "debug"
if os.environ.get('name'):
    name = os.environ.get('name')
files = name + '.files/'
cycle = os.environ.get('CYCLE')
if cycle == '':
    cycle = 1
else:
    cycle = int(cycle)
if cycle == 1:    
    pass
    #print("  tf", tf.__version__)
    #print("  eager execution:", tf.executing_eagerly())
print("Starting cycle", cycle)


# Try to make results reproducible.
if not 'PYTHONHASHSEED' in os.environ:
    os.environ['PYTHONHASHSEED'] = "832"
seed = 42
random.seed(seed)
set_random_seed(seed)
np.random.seed(seed)
print("  Random seeds:")
print("      python hash", os.environ['PYTHONHASHSEED'])
print("      random     ", seed)
print("      np.random  ", seed)
print("      tf         ", seed)

if cycle == 1:
    print("  # atom types:", n_atom_types)
    print("  # edge features:", n_edge_features)
    print("  hidden state dimension:", state_dim)
    print("  output regressor units (max):", reg_dim)
    print("  MP time steps:", T)
    print("  learning_rate:", learning_rate)
    print("  batch_size:", batch_size)
    print("  epochs:", epochs)


# Load data.
tic = time()

if os.path.exists(picklefile):
    print("Loading training data from", picklefile)
    input_data, y_train, y_dev = pickle.load(open(picklefile, 'rb'))
else:
    input_data, y_train, y_dev = load_data(datadir)
    pickle.dump([input_data, y_train, y_dev], open(picklefile, 'wb'))

print("Boarding complete after", time() - tic); sys.stdout.flush()


# Construct input sequences
train_sequence = RBFSequence(input_data['inputs_train'], y_train, batch_size)
valid_sequence = RBFSequence(input_data['inputs_valid'], y_dev, batch_size)
print("Batch size:", batch_size)
print("Number of batches:", len(train_sequence), len(valid_sequence))

# Shuffle data
train_sequence.on_epoch_end()


# Raw (integer) graph inputs
node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
atom_types = Input(shape=(1,), name='atom', dtype='int32')
edge_features = Input(shape=(42,), name='bond', dtype='float32')
distance_rbf = Input(shape=(150,), name='distance_rbf', dtype='float32')
connectivity = Input(shape=(2,), name='connectivity', dtype='int32') 

# 1D inputs -> scalars
#snode_graph_indices = Squeeze(name='squeeze_node_graph_indices')(node_graph_indices)   # only needed for atomwise_energy
satom_types = Squeeze(name='squeeze_atom_types')(atom_types)

# Initialize the atom states
atom_state = Embedding(
    n_atom_types,
    state_dim, name='atom_embedding')(satom_types)
#print("atom_state:", atom_state.shape)

# Skip this feature
#atomwise_energy = Embedding(
#    n_atom_types, 1, name='atomwise_energy',
#    embeddings_initializer=keras.initializers.constant(atom_contributions.values)
#)(satom_types)

# Try shared embedding + BN of edge features
edge_embedding = Dense(state_dim*2, activation='softplus', name='edge_embedding_FC1')(edge_features)
edge_embedding = BatchNormalization(name='edge_embedding_BN1')(edge_embedding)
edge_embedding = Dense(state_dim, activation='softplus', name='edge_embedding_FC2')(edge_embedding)

# Atom pair Gaussians + edge_features
bond_state = Concatenate(name='cat_rbf_edge_features')([distance_rbf, edge_embedding])
#bond_state = distance_rbf

def message_block(atom_state, bond_state, connectivity):

    source_atom = GatherAtomToBond(1, name=f'source_atom_{t}')([atom_state, connectivity])
    target_atom = GatherAtomToBond(0, name=f'target_atom_{t}')([atom_state, connectivity])
    #print("source_atom, target_atom:", source_atom.shape, target_atom.shape)   # (m, state_dim)

    # Edge update network
    bond_state = Concatenate(name=f'edge_update_concat_{t}')([source_atom, target_atom, bond_state])
    bond_state = Dense(2*state_dim, activation='softplus', name=f'edge_update_FC1_{t}')(bond_state)
    bond_state = Dense(state_dim, name=f'edge_update_FC2_{t}')(bond_state)

    # message function
    bond_state = Dense(state_dim, activation='softplus', name=f'edge_update_FC3_{t}')(bond_state)
    bond_state = Dense(state_dim, activation='softplus', name=f'edge_update_FC4_{t}')(bond_state)

    source_atom = Dense(state_dim, name=f'message_embed_source_atom_{t}')(source_atom)
    messages = Multiply(name=f'message_multiply_{t}')([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer='sum', name=f'message_aggregate_{t}')([messages, connectivity])
    
    # state transition function
    messages = Dense(state_dim, activation='softplus', name=f'message_FC1_{t}')(messages)
    messages = Dense(state_dim, name=f'message_FC2_{t}')(messages)
    atom_state = Add(name=f'node_update_{t}')([atom_state, messages])
    
    return atom_state, bond_state

for t in range(T):
    atom_state, bond_state = message_block(atom_state, bond_state, connectivity)


# Enumerate the edges in the batch for bookkeeping.
edge_indices = Lambda(lambda x: K.reshape(K.arange(K.shape(x)[0], dtype=K.floatx()), (-1, 1)))(edge_features)

# Concat all tensors with edge_features to split them into "1JHC", "1JHN", "otherJ", and "rest" bond types.
# Integer Tensors need to be casted to float for this.
connectivity_float = Lambda(lambda x: tf.cast(x, tf.float32))(connectivity)
concat = Concatenate(name='ER5_concat')([edge_features, edge_indices, connectivity_float, bond_state])
concat_1JHC = Lambda(split_1JHC)(concat)
concat_1JHN = Lambda(split_1JHN)(concat)
concat_otherJ = Lambda(split_otherJ)(concat)
concat_rest = Lambda(split_rest)(concat)

# Undo the concat to get back the splitted original tensors.
get_edge_features = Lambda(lambda x: x[:, 0:n_edge_features])
get_edge_index = Lambda(lambda x: x[:, n_edge_features:n_edge_features+1])  # need shape(m, 1) not (m,)
get_connectivity = Lambda(lambda x: tf.cast(x[:, n_edge_features+1:n_edge_features+1+2], tf.int32))
get_bond_state = Lambda(lambda x: x[:, n_edge_features+1+2:n_edge_features+1+2+state_dim])

edge_features_1JHC = get_edge_features(concat_1JHC)
edge_features_1JHN = get_edge_features(concat_1JHN)
edge_features_otherJ = get_edge_features(concat_otherJ)
edge_features_rest = get_edge_features(concat_rest)

edge_index_1JHC = get_edge_index(concat_1JHC)
edge_index_1JHN = get_edge_index(concat_1JHN)
edge_index_otherJ = get_edge_index(concat_otherJ)
# for the irrelevant bonds with no labels just need their original indices.
edge_index_rest = get_edge_index(concat_rest)

connectivity_1JHC = get_connectivity(concat_1JHC)
connectivity_1JHN = get_connectivity(concat_1JHN)
connectivity_otherJ = get_connectivity(concat_otherJ)
connectivity_rest = get_connectivity(concat_rest)

bond_state_1JHC = get_bond_state(concat_1JHC)
bond_state_1JHN = get_bond_state(concat_1JHN)
bond_state_otherJ = get_bond_state(concat_otherJ)
bond_state_rest = get_bond_state(concat_rest)

# Now feed the split tensors to different Readout functions.
output_1JHC = ERX(edge_features_1JHC, atom_state, connectivity_1JHC, bond_state_1JHC, "1JHC", reg_dim)
output_1JHN = ERX(edge_features_1JHN, atom_state, connectivity_1JHN, bond_state_1JHN, "1JHN", reg_dim)
output_otherJ = ERX(edge_features_otherJ, atom_state, connectivity_otherJ, bond_state_otherJ, "otherJ", reg_dim)

# Simulate output for the irrelevant bonds without labels.
output_rest = Lambda(rest_faker)(edge_index_rest)
#output_rest = Lambda(rest_faker)(bond_state_rest)

# Concatenate the outputs with the corresponding part of the edge index.
output_1JHC = Concatenate()([output_1JHC, edge_index_1JHC])
output_1JHN = Concatenate()([output_1JHN, edge_index_1JHN])
output_otherJ = Concatenate()([output_otherJ, edge_index_otherJ])
output_rest = Concatenate()([output_rest, edge_index_rest])

# Restore the original order of the bonds and strip off the indices again.
output = Lambda(idx_join)([output_1JHC, output_1JHN, output_otherJ, output_rest])


# Create the model, compile, go for coffee.
model = GraphModel(inputs=[node_graph_indices, atom_types, edge_features, 
                           distance_rbf, connectivity],
                   outputs=[output])

if transfer_mp:
    # Before compiling: freeze layers except Readout block
    for layer in model.layers[0:54]:  # fit also last two edge update layers
        layer.trainable = False

model.compile(optimizer=Adam(lr=learning_rate), loss=masked_mae_1d)

print("Checking the list of layers:")
for i, layer in enumerate(model.layers):
    print(i, layer.name)
print()
model.summary()


if not os.path.exists(files):
    os.makedirs(files)
checkpoint = ModelCheckpoint(files + "best_model.hdf5", save_best_only=False, period=chk_period, verbose=2)
csv_logger = CSVLogger(files + 'log.csv')
print("Logfile and checkpoints written in", files)

lr_decay = LearningRateScheduler(decay_fn)

# Load the weights from a HDF5 file (all layers with matching names).
h5file = name + "-weights.h5"
try:
    model.load_weights(h5file, by_name=transfer_mp)  # ignore layer names if transfer_mp == False
    print("Model weights loaded from", h5file)
except OSError:
    print("Starting from scratch")
    if transfer_mp:
        raise OSError("Need weights for transfer learning")
    pass

sys.stdout.flush()

hist = model.fit_generator(generator=train_sequence, epochs=epochs,
                           validation_data=valid_sequence, validation_freq=val_period,
                           callbacks=[checkpoint, csv_logger, lr_decay],
                           verbose=2) #, use_multiprocessing=True, workers=8)

# Save the weights to a HDF5 file.
model.save_weights(h5file)
print("Wrote weights to", h5file)
