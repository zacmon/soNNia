#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 Montague, Zachary
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Script containing useful functions for sequence generation using righor
"""
from __future__ import annotations
from itertools import chain, repeat
import multiprocessing as mp
import os
from typing import *

import numpy as np
from numpy.typing import ArrayLike, NDArray
import righor as rg
from tqdm import tqdm

from sonnia.utils import HEAVY_CHAINS, LIGHT_CHAINS, get_model_dir

def record_cdr3aa_v_j_cdr3nt(
    gen_seq: rg.GenerationResult
) -> Tuple[str]:
    """
    Collect the CDR3 amino acid sequence, V gene, J gene, and CDR3 nucleotide sequence
    from a generated recombination event.

    Parameters
    ----------
    gen_seq : righor.GenerationResult
        A generated event from righor.

    Returns
    -------
    tuple of str
        The CDR3 amino acid sequence, V gene, J gene, and CDR3 nucleotide sequence.
    """
    return (
        gen_seq.cdr3_aa, gen_seq.v_gene.partition('*')[0],
        gen_seq.j_gene.partition('*')[0], gen_seq.cdr3_nt
    )

def oof_seq_generator(
    func: Callable[bool, rg.GenerationResult],
    num_monte_carlo: int,
) -> Generator[rg.GenerationResult, None, None]:
    """
    Generator for producing out-of-frame events.

    Parameters
    ----------
    func : callable
        A function for generation sequences from the righor generator object.
    num_monte_carlo : int
        The total number of sequences to produce.

    Yields
    ------
    gen_seq : righor.GenerationResult
        A generated recombination event.
    """
    oof_seqs_seen = 0
    while oof_seqs_seen < num_monte_carlo:
        gen_seq = func(functional=False)
        if gen_seq.cdr3_aa is not None:
            continue
        yield gen_seq
        oof_seqs_seen += 1

def target_filter_oof_seq_generator(
    func: Callable,
    num_monte_carlo: int,
    filter_func: Callable[rg.GenerationResult, bool],
) -> Generator[rg.GenerationResult, None, None]:
    """
    Generator for producing targeted filtered out-of-frame events.

    Parameters
    ----------
    func : callable
        A function for generation sequences from the righor generator object.
    num_monte_carlo : int
        The total number of sequences that will be retained after filtering.
        The ensuing amount of events retained will be equal to num_monte_carlo.
    filter_func : callback, optional
        A function which takes in a GenerationResult and returns a bool to filter
        which generated sequences are retained. I.e., sequences which result
        in a value of True for this function are kept.

    Yields
    ------
    gen_seq : righor.GenerationResult
        A generated recombination event.
    """
    num_target_seqs = 0
    while num_target_seqs < num_monte_carlo:
        gen_seq = func(functional=False)
        if gen_seq.cdr3_aa is not None:
            continue
        if not filter_func(gen_seq):
            continue
        yield gen_seq
        num_target_seqs += 1

def target_filter_seq_generator(
    func: Callable,
    num_monte_carlo: int,
    functional: bool,
    filter_func: Callable[rg.GenerationResult, bool],
) -> Generator[rg.GenerationResult, None, None]:
    """
    Generator for producing targeted filtered events.

    Parameters
    ----------
    func : callable
        A function for generation sequences from the righor generator object.
    num_monte_carlo : int
        The total number of sequences that will be retained after filtering.
        The ensuing amount of events retained will be equal to num_monte_carlo.
    functional : bool
        Specify if all the events should be functional, i.e., no stop codons
        and in-frame.
    filter_func : callback, optional
        A function which takes in a GenerationResult and returns a bool to filter
        which generated sequences are retained. I.e., sequences which result
        in a value of True for this function are kept.

    Yields
    ------
    gen_seq : righor.GenerationResult
        A generated recombination event.
    """
    num_target_seqs = 0
    while num_target_seqs < num_monte_carlo:
        gen_seq = func(functional)
        if not filter_func(gen_seq):
            continue
        num_target_seqs += 1
        yield gen_seq

def seq_gen_worker(
    igor_model_files: Sequence[str],
    num_monte_carlo: int,
    model_type: str = 'vdj',
    seed: Optional[int] = None,
    available_v: Optional[Iterable] = None,
    available_j: Optional[Iterable] = None,
    functional: bool = True,
    out_of_frame_only: bool = False,
    without_error: bool = True,
    container_func: Callback[List[List[str]], Any] = np.array,
    mc_type: str = 'total',
    filter_func: Callback[rg.GenerationResult, bool] = None,
    record_func: Callback[rg.GenerationResult, Any] = record_cdr3aa_v_j_cdr3nt,
    verbose: bool = False,
) -> Any:
    """
    Generate productive recombinations using the righor model.

    This is a separate function for easy parallelization.

    Parameters
    ----------
    igor_model_files : sequence of str
        A list of files required by righor to specify the model.
    num_monte_carlo : int
        The number of sequences to generate.
    model_type : str, default 'vdj'
        Specifies whether the model is VDJ or VJ.
    seed : int, optional
        The seed for the sequence generator.
    available_v : iterable of str, optional
        The V genes used to produce recombinations.
        If None, there is no restriction on the V genes.
    available_j : iterable of str, optional
        The J genes used to produce recombinations.
        If None, there is no restriction on the J genes.
    functional : bool, default True
        If True, return sequences only from the productive repertoire.
    out_of_frame_only : bool, default False
        If True, return sequences which are out-of-frame only.
    without_errors : bool, default True
        Generate sequences without introducing errors.
    container_func : callback, default np.array
        A function initializing the container for the resulting sequences.
        If None, the list of sequences will be returned.
    mc_type : str, default 'total'
        Gives whether the num_monte_carlo specified is how many sequences are
        generated in total ('total') or how many sequences are desired to be
        retained after filtering ('target'). This has an effect only if filter_func
        is provided.
    filter_func : callback, optional
        A function which takes in a GenerationResult and returns a bool to filter
        which generated sequences are retained. I.e., sequences which result
        in a value of True for this function are kept.
    record_func : callback, default record_cdr3aa_v_j_cdr3nt
        A function which takes a GenerationResult and returns what information
        from the GenerationResult would like to be kept. By default,
        a function, record_cdr3aa_v_j_cdr3nt, is used which records the CDR3 amino acid
        sequence, V gene, J gene, and CDR3 nucleotide sequence.
    verbose : bool, default False
        Display progress bar.

    Returns
    -------
    container of str
        Monte Carlo sequences produced from VDJ recombination. The default
        container is a numpy.ndarray.
    """
    if model_type not in {'vdj', 'vj'}:
        raise ValueError('model_type must be \'vdj\' or \'vj\'.')

    if mc_type not in {'total', 'target'}:
        raise ValueError('mc_type must be \'total\' or \'target\'.')

    igor_model = getattr(rg, model_type).Model.load_model_from_files(*igor_model_files)

    if out_of_frame_only:
        functional = False

    functional_str = ''
    if functional:
        igor_model.v_segments = [
            v for v in igor_model.v_segments if v.functional in {'F', '(F)'}
        ]
        igor_model.j_segments = [
            j for j in igor_model.j_segments if j.functional in {'F', '(F)'}
        ]
        functional_str = 'functional '

    if available_v is not None:
        available_v = set(available_v)
        available_v = [
            v for v in igor_model.v_segments if v.name.partition('*')[0] in available_v
        ]
    if available_j is not None:
        available_j = set(available_j)
        available_j = [
            j for j in igor_model.j_segments if j.name.partition('*')[0] in available_j
        ]

    generator = igor_model.generator(seed, available_v, available_j)
    if without_error:
        func = generator.generate_without_errors
    else:
        func = generator.generate

    seqs = []

    if filter_func is None:
        if out_of_frame_only:
            for gen_seq in tqdm(
                oof_seq_generator(func, num_monte_carlo), position=0,
                total=num_monte_carlo, disable=not verbose, desc='Generting oof seqs'
            ):
                seqs.append(record_func(gen_seq))
        else:
            for _ in tqdm(
                range(num_monte_carlo), position=0, disable=not verbose,
                desc=f'Generating {functional_str}seqs',
            ):
                gen_seq = func(functional=functional)
                seqs.append(record_func(gen_seq))

    else:
        if mc_type == 'total':
            if out_of_frame_only:
                for gen_seq in tqdm(
                    oof_seq_generator(func, num_monte_carlo),
                    position=0, disable=not verbose, total=num_monte_carlo,
                    desc='Generating filtered oof seqs'
                ):
                    if filter_func(gen_seq):
                        seqs.append(record_func(gen_seq))
            else:
                for _ in tqdm(
                    range(num_monte_carlo), position=0, disable=not verbose,
                    desc=f'Generating filtered {functional_str}seqs',
                ):
                    gen_seq = func(functional=functional)
                    if not filter_func(gen_seq):
                        continue
                    seqs.append(record_func(gen_seq))
        else:
            num_target_seqs = 0
            if out_of_frame_only:
                for gen_seq in tqdm(
                    target_filter_oof_seq_generator(func, num_monte_carlo, filter_func),
                    position=0, disable=not verbose, total=num_monte_carlo,
                    desc='Generating targeted oof seqs'
                ):
                    seqs.append(record_func(gen_seq))
            else:
                for gen_seq in tqdm(
                    target_filter_seq_generator(func, num_monte_carlo, functional, filter_func),
                    position=0, disable=not verbose, total=num_monte_carlo,
                    desc=f'Generating targeted {functional_str}seqs'
                ):
                    seqs.append(record_func(gen_seq))

    if container_func is None:
        return seqs
    else:
        return container_func(seqs)

def get_chunks(
    data_size: int,
    num_chunks: int
) -> List[int]:
    """
    Split size of data into a certain number of chunks.

    Parameters
    ----------
    data_size : int
        The size of the data to be chunked.
    num_chunks : int
        The desired number of chunks.

    Returns
    -------
    chunks : list of int
        A list containing the sizes of the chunks.
    """
    remainder = int(data_size % num_chunks)
    chunk_size = data_size // num_chunks
    chunks = [chunk_size] * (num_chunks - remainder) + [chunk_size + 1] * remainder
    return chunks

def get_child_seeds(
    num_children: int,
    seed: int | ArrayLike | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
) -> NDArray[np.int64]:
    """
    Generate seeds for child processes in a way consistent with righor.

    Parameters
    ----------
    num_children : int
        The number of seeds to be produced.
    seed : int or array_like of int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        The initial seed used to generate child seeds.

    Returns
    -------
    numpy.ndarray
        An array containing the children's seeds.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(2**32 - 1, size=num_children)

def get_igor_model_files(
    model: str | Sonia | SoNNia,
) -> List[str]:
    """
    Return the files necessary for initializing a righor object.

    Parameters
    ----------
    model : str or Sonia or SoNNia
        If str, model points to a directory containing the Pgen model files.
        Otherwise, model is a Sonia object with the pgen_dir attribute.

    Returns
    -------
    list of str
        A list containing strings pointing to the files required by righor.
    """
    if isinstance(model, str):
        model = get_model_dir(model)
        return [
            os.path.join(model, 'model_params.txt'),
            os.path.join(model, 'model_marginals.txt'),
            os.path.join(model, 'V_gene_CDR3_anchors.csv'),
            os.path.join(model, 'J_gene_CDR3_anchors.csv')
        ]

    return [
        os.path.join(model.pgen_dir, 'model_params.txt'),
        os.path.join(model.pgen_dir, 'model_marginals.txt'),
        os.path.join(model.pgen_dir, 'V_gene_CDR3_anchors.csv'),
        os.path.join(model.pgen_dir, 'J_gene_CDR3_anchors.csv')
    ]

def generate_pgen_seqs(
    model: str | Sonia | SoNNia,
    num_monte_carlo: int,
    seed: int | ArrayLike | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
    available_v: Optional[Iterable] = None,
    available_j: Optional[Iterable] = None,
    functional: bool = True,
    out_of_frame_only: bool = False,
    without_error: bool = True,
    container_func: Callback[List[List[str]], Any] = np.array,
    concat_func: Callback[Any, Any] = np.concatenate,
    mc_type: str = 'total',
    filter_func: Callback[rg.GenerationResult, bool] = None,
    record_func: Callback[rg.GenerationResult, Any] = record_cdr3aa_v_j_cdr3nt,
    processes: Optional[int] = None
) -> Any:
    """
    Generate sequences from the recombination using CPU parallelization.

    Parameters
    ----------
    model : str or Sonia or SoNNia
        A string pointing to the directory containing the  model files or a Sonia/SoNNia model object.
    num_monte_carlo : int
        The number of sequences to generate.
    seed : int or array_like of int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        The initial seed used to generate child seeds.
    available_v : iterable of str, optional
        The V genes used to produce recombinations.
        If None, there is no restriction on the V genes.
    available_j : iterable of str, optional
        The J genes used to produce recombinations.
        If None, there is no restriction on the J genes.
    functional : bool, default True
        If True, return sequences only from the productive repertoire.
    out_of_frame_only : bool, default False
        If True, return sequences which are out-of-frame only.
    without_errors : bool, default True
        Generate sequences without introducing errors.
    container_func : callback, default np.array
        A function initializing the container for the resulting sequences.
        If None, the list of sequences will be returned.
    concat_func : callback, default np.concatenate
        A function that will combine the containers of type returned by
        container_func. This is used only when parallelization is enabled.
    mc_type : str, default 'total'
        Gives whether the num_monte_carlo specified is how many sequences are
        generated in total ('total') or how many sequences are desired to be
        retained after filtering ('target'). This has an effect only if filter_func
        is provided.
    filter_func : callback, optional
        Function which takes in a GenerationResult and returns a bool to filter
        which generated sequences are retained. I.e., sequences which result
        in a value of True for this function are kept.
    record_func : callback, default record_cdr3aa_v_j_cdr3nt
        A function which takes a GenerationResult and returns what information
        from the GenerationResult would like to be kept. By default,
        a function, record_cdr3aa_v_j_cdr3nt, is used which records the CDR3 amino acid
        sequence, V gene, J gene, and CDR3 nucleotide sequence.
    processes : int, optional
        The number of CPU cores over which generation will be parallelized.
        If None, the maximum number of CPUs is used.

    Returns
    -------
    container of str
        Monte Carlo sequences produced from VDJ recombination. The default
        container is a numpy.ndarray.
    """
    if processes is not None:
        if not isinstance(processes, (int, np.integer)):
            raise TypeError('processes must be an integer.')
        if processes <= 0:
            raise ValueError('processes must be >= 1.')

    if filter_func is not None:
        if not callable(filter_func):
            raise ValueError('filter_func must be callable.')

    if mc_type not in {'total', 'target'}:
        raise ValueError('mc_type must be \'total\' or \'target\'.')

    igor_model_files = get_igor_model_files(model)

    # Peek inside the J anchor file to determine if the model is VDJ or VJ.
    with open(igor_model_files[-1], 'r') as fin:
        # Skip headers.
        next(fin)

        # Get chain.
        receptor_chain = next(fin).partition('J')[0]

    if receptor_chain in HEAVY_CHAINS:
        model_type = 'vdj'
    elif receptor_chain in LIGHT_CHAINS:
        model_type = 'vj'
    else:
        heavy_chain_str = f'{HEAVY_CHAINS}'[1:-1]
        light_chain_str = f'{LIGHT_CHAINS}'[1:-1]
        raise RuntimeError(
            f'Unrecognized chain: {receptor_chain}. Recognized heavy chains: '
            f'{heavy_chain_str} Recognized light chains: {light_chain_str}.'
        )

    if processes is None:
        processes = os.cpu_count()

    if num_monte_carlo <= int(5e5) and mc_type == 'total':
        processes = 1


    if processes == 1:
        seed = get_child_seeds(1, seed)[0]
        return seq_gen_worker(
            igor_model_files, num_monte_carlo, model_type, seed, available_v,
            available_j, functional, out_of_frame_only, without_error,
            container_func, mc_type, filter_func, record_func, verbose=True
        )

    if filter_func is not None and getattr(filter_func, '__name__') == '<lambda>':
        raise TypeError(
            'filter_func must not be a lambda function in order to be pickled '
            'correctly for parallelization. Please rewrite filter_func as a named '
            'function, e.g., def filter_func(seq): ...'
        )

    if container_func is not None and getattr(container_func, '__name__') == '<lambda>':
        raise TypeError(
            'container_func must not be a lambda function in order to be pickled '
            'correctly for parallelization. Please rewrite container_func as a named '
            'function, e.g., def filter_func(seq): ...'
        )

    if container_func is not None and concat_func is None:
        raise ValueError(
            'concat_func must not be None if container_func is not None. '
            'Please specify a function for concat_func so the output can be '
            'combined correctly.'
        )

    if getattr(record_func, '__name__') == '<lambda>':
        raise TypeError(
            'record_func must not be a lambda function in order to be pickled '
            'correctly for parallelization. Please rewrite record_func as a named '
            'function, e.g., def filter_func(seq): ...'
        )

    seeds = get_child_seeds(processes, seed)
    chunks = get_chunks(num_monte_carlo, processes)

    zipped = zip(
        repeat(igor_model_files), chunks, repeat(model_type), seeds, repeat(available_v),
        repeat(available_j), repeat(functional), repeat(out_of_frame_only),
        repeat(without_error), repeat(container_func), repeat(mc_type),
        repeat(filter_func), repeat(record_func)
    )

    with mp.Pool() as pool:
        gen_seqs = pool.starmap(seq_gen_worker, zipped)

    if container_func is None:
        return list(chain.from_iterable(gen_seqs))
    return concat_func(gen_seqs)
