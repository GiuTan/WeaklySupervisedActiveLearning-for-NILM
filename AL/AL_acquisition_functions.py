"""
@Author: Tamara Sobot
@Date: December 2023
"""

import numpy as np


def random_query(generator, args):
    """
    Randomly query samples from query pool.
    """
    mask = np.random.choice(np.arange(generator.query_pool_indices.size),
                            min(args.queries_per_iteration, generator.query_pool_indices.size), replace=False)

    new_train_indexes = generator.query_pool_indices[mask]
    new_validate_indexes = np.delete(generator.query_pool_indices, mask)

    return new_train_indexes, new_validate_indexes


def pool_based_uncertainty(model, generator, args, al_iteration):
    """
    Query samples from query pool based on uncertainty.
    """

    output = model.predict(x=generator.x[generator.query_pool_indices])

    # adjust for the special case.
    confidence = np.abs(output - args.output_threshold)

    mean_confidence = np.mean(confidence, axis=-1)
    mean_confidence = np.squeeze(mean_confidence)

    mask = np.argpartition(mean_confidence, min(args.queries_per_iteration, len(mean_confidence)-1))[:min(args.queries_per_iteration, len(mean_confidence))]

    new_train_indexes = generator.query_pool_indices[mask]
    new_validate_indexes = np.delete(generator.query_pool_indices, mask)

    return new_train_indexes, new_validate_indexes


def whole_pool(generator):
    """
    Query the whole query pool at once.
    """
    return generator.query_pool_indices, []
