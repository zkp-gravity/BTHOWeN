#!/usr/bin/false

import numpy as np
from numba import jit

from bloom_filter import BloomFilter, h3_hash

def export_to_file(fname, yv):
    with open(fname, 'w') as f:
        print('{', file=f)
        for i in range(yv.shape[0]):
            print(f'    in{i}: {{', file=f)
            in_i = ',\n'.join(f'        bit{j}: {yv[i,j]}field' for j in range(yv.shape[1]))
            print(in_i, file=f)
            print('    },' if i < yv.shape[0]-1 else '    }', file=f)
        print('}', file=f)

# Converts a vector of booleans to an unsigned integer
#  i.e. (2**0 * xv[0]) + (2**1 * xv[1]) + ... + (2**n * xv[n])
# Inputs:
#  xv: The boolean vector to be converted
# Returns: The unsigned integer representation of xv
@jit(nopython=True, inline='always')
def input_to_value(xv):
    result = 0
    for i in range(xv.size):
        result += xv[i] << i
    return result

# Generates a matrix of random values for use as m-arrays for H3 hash functions
def generate_h3_values(num_inputs, num_entries, num_hashes):
    assert(np.log2(num_entries).is_integer())
    shape = (num_hashes, num_inputs)
    values = np.random.randint(0, num_entries, shape)
    return values

# Implementes a single discriminator in the WiSARD model
# A discriminator is a collection of boolean LUTs with associated input sets
# During inference, the outputs of all LUTs are summed to produce a response
class Discriminator:
    # Constructor
    # Inputs:
    #  num_inputs:    The total number of inputs to the discriminator
    #  unit_inputs:   The number of boolean inputs to each LUT/filter in the discriminator
    #  unit_entries:  The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:   The number of hash functions for each filter.
    #  random_values: If provided, is used to set the random hash seeds for all filters. Otherwise, each filter generates its own seeds.
    def __init__(self, num_inputs, unit_inputs, unit_entries, unit_hashes, random_values=None):
        assert((num_inputs/unit_inputs).is_integer())
        self.num_filters = num_inputs // unit_inputs
        self.filters = [BloomFilter(unit_inputs, unit_entries, unit_hashes, random_values) for i in range(self.num_filters)]

    # Performs a training step (updating filter values)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1) # Divide the inputs between the filters
        for idx, inp in enumerate(filter_inputs):
            self.filters[idx].add_member(inp)

    # Performs an inference to generate a response (number of filters which return True)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: The response of the discriminator to the input
    def predict(self, xv, yv, i):
        filter_inputs = xv.reshape(self.num_filters, -1) # Divide the inputs between the filters
        response = 0
        for idx, inp in enumerate(filter_inputs):
            # len(inp) == 28
            res, ds = self.filters[idx].check_membership(inp)
            yv[idx, i] = ds[0]
            yv[idx, i + 10] = ds[1] # 10 is the number of discriminators
            response += int(res)
        return response
    
    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for f in self.filters:
            f.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for f in self.filters:
            f.binarize()
    
# Top-level class for the WiSARD weightless neural network model
class WiSARD:
    # Constructor
    # Inputs:
    #  num_inputs:       The total number of inputs to the model
    #  num_classes:      The number of distinct possible outputs of the model; the number of classes in the dataset
    #  unit_inputs:      The number of boolean inputs to each LUT/filter in the model
    #  unit_entries:     The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:      The number of hash functions for each filter.
    def __init__(self, num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes):
        self.pad_zeros = (((num_inputs // unit_inputs) * unit_inputs) - num_inputs) % unit_inputs
        pad_inputs = num_inputs + self.pad_zeros
        self.input_order = np.arange(pad_inputs) # Use each input exactly once
        np.random.shuffle(self.input_order) # Randomize the ordering of the inputs
        random_values = generate_h3_values(unit_inputs, unit_entries, unit_hashes)
        self.discriminators = [Discriminator(self.input_order.size, unit_inputs, unit_entries, unit_hashes, random_values) for i in range(num_classes)]

    # Performs a training step (updating filter values) for all discriminators
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv, label):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order] # Reorder input
        self.discriminators[label].train(xv)

    # Performs an inference with the provided input
    # Passes the input through all discriminators, and returns the one or more with the maximal response
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: A vector containing the indices of the discriminators with maximal response
    def predict(self, xv):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order] # Reorder input

        # Each discriminator has 56 filters
        num_filters = 56
        yv = xv.reshape(num_filters, 28)
        export_to_file('input_file.txt', yv)

        filter_bits = 10 # 1024
        num_hashes = 2
        hashes1 = [h3_hash(yv[i], num_hashes, filter_bits) for i in range(yv.shape[0])]
        hashes2 = np.array([[h[0], h[1], b] for (h, b) in hashes1], dtype=np.int64)
        quot = 7
        with open('hash_values.txt', 'w') as f:
            print('{', file=f)
            for i in range(hashes2.shape[0]):
                h0, h1, msb = hashes2[i]
                h = h0 + (h1<<10) + (msb<<20)

                yy = yv[i]
                x = int(0)
                for i in range(len(yy)):
                    x += int(yy[i] << i)
                x3 = x * x * x
                p = 2097143
                quot = (x3 - h) // p
                assert x3 == quot * p + h, f'{x3}, {quot}, {h}, {x3 % p}'
                assert x3 % p == h, f'{h} != {x3 % p}'

                print(f'    info{i}: {{', file=f)
                print(f'        decomposition{i}: {{', file=f)
                print(f'            index1: {h0}field', file=f)
                print(f'            index2: {h1}field', file=f)
                print(f'            msb: {msb}field', file=f)
                print(f'        }},', file=f)
                print(f'        hash: {h}field,', file=f)
                print(f'        quotient: {quot}field', file=f)
                print('    },' if i < hashes2.shape[0]-1 else '    }', file=f)
            print('}', file=f)

        yv = np.zeros((num_filters, num_hashes * len(self.discriminators)), dtype=np.int64)

        responses = np.array([d.predict(xv, yv, i) for i, d in enumerate(self.discriminators)], dtype=int)
        export_to_file('bloom_filters.txt', yv)

        max_response = responses.max()
        winner = np.where(responses == max_response)[0]

        with open('winning_discriminator_index.txt', 'w') as f:
            print(f'{{ winner: {winner[0]}field }}', file=f)

        with open('winning_discriminator_value.txt', 'w') as f:
            print(f'{{ max_response: {max_response}field }}', file=f)

        return winner

    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for d in self.discriminators:
            d.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for d in self.discriminators:
            d.binarize()
   
