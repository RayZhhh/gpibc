from __future__ import annotations

import random
from typing import List, Tuple

import fset
import tree
from tree import Node, TreeGenerator


def _subtree_index(prefix: List[Node], start_pos) -> Tuple[int, int]:
    func_count = 0
    term_count = 0
    end = start_pos
    while end < len(prefix):
        node = prefix[end]
        if node.is_binary_function():
            func_count += 1
        elif node.is_terminal():
            term_count += 1
        if func_count + 1 == term_count:
            break
        end += 1
    return start_pos, end + 1


class Program:
    def __init__(self, img_h, img_w, init_depth=6, init_method=None, prefix=None):
        self.img_h = img_h
        self.img_w = img_w
        self.prefix: List[Node] = []
        self.depth = 0
        self.fitness = 0
        self._init_depth = init_depth
        self._init_method = init_method

        # init by prefix
        if prefix is not None:
            self.prefix = prefix
            self.get_depth_of_program()
            return

        # init respect  to the init method
        if init_method is not None:
            if init_method == 'full':
                gener = TreeGenerator(depth=init_depth, img_h=img_h, img_w=img_w)
                self.prefix = gener.full_init_tree()
                self.get_depth_of_program()
            elif init_method == 'growth':
                gener = TreeGenerator(depth=init_depth, img_h=img_h, img_w=img_w)
                self.prefix = gener.growth_init_tree()
                self.get_depth_of_program()
            else:
                raise RuntimeError("init_method must be 'full' or 'growth'.")

    def __repr__(self):
        ret = '[ '
        for node in self.prefix:
            ret += str(node) + ' '
        ret += ']'
        return ret

    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, item):
        return self.prefix[item]

    def get_depth_of_program(self):
        s = []
        for node in reversed(self.prefix):
            if node.is_terminal():
                s.append(1)
            elif node.is_binary_function():
                depth0 = s.pop()
                depth1 = s.pop()
                s.append(max(depth0, depth1) + 1)
            else:
                depth = s.pop()
                s.append(depth + 1)
        self.depth = s.pop()

    def crossover(self, donor: Program | List[Node]):
        if isinstance(donor, Program):
            dprefix = donor.prefix
        else:
            dprefix = donor
        self_start = random.randint(1, len(self.prefix) - 1)
        self_start, self_end = _subtree_index(prefix=self.prefix, start_pos=self_start)

        root_func = self.prefix[self_start].name
        if root_func == fset.Sub:
            root_indexes = [i for i in range(len(donor)) if donor[i].name == fset.Sub or donor[i].name == fset.G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            donor_start, donor_end = _subtree_index(prefix=dprefix, start_pos=rand_start_pos)
        elif root_func == fset.G_Std:
            root_indexes = [i for i in range(len(donor)) if donor[i].name == fset.G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            donor_start, donor_end = _subtree_index(prefix=dprefix, start_pos=rand_start_pos)
        else:
            root_indexes = [i for i in range(len(donor)) if donor[i].name != fset.Sub and donor[i].name != fset.G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            donor_start, donor_end = _subtree_index(prefix=dprefix, start_pos=rand_start_pos)

        # gen new prefix
        ret_prefix = [self[i] for i in range(0, self_start)]
        ret_prefix += [donor[i] for i in range(donor_start, donor_end)]
        ret_prefix += [self[i] for i in range(self_end, len(self))]

        self.prefix = ret_prefix
        self.get_depth_of_program()

    def point_mutation(self):
        point_indexes = [i for i in range(len(self.prefix))
                         if self.prefix[i].name != fset.Sub and self.prefix[i].name != fset.G_Std]
        pos = random.sample(population=point_indexes, k=1)[0]

        if self[pos].is_terminal():
            self.prefix[pos] = tree.rand_terminal_node(self.img_h, self.img_w)
        else:
            self.prefix[pos] = tree.rand_inter_func_node()

    def subtree_mutation(self):
        gener = TreeGenerator(depth=self._init_depth, img_h=self.img_h, img_w=self.img_w)
        if self._init_method == 'full':
            rand_tree = gener.full_init_tree()
        elif self._init_method == 'growth':
            rand_tree = gener.growth_init_tree()
        else:
            if random.random() < 0.5:
                rand_tree = gener.full_init_tree()
            else:
                rand_tree = gener.growth_init_tree()
        return self.crossover(rand_tree)

    def hoist_mutation(self):
        # random subtree of self.prefix
        root_indexes = [i for i in range(1, len(self.prefix)) if self.prefix[i].name != fset.G_Std]
        rand_start_pos = random.sample(population=root_indexes, k=1)[0]
        start1, end1 = _subtree_index(prefix=self.prefix, start_pos=rand_start_pos)
        subtree = [self.prefix[i] for i in range(start1, end1)]

        if self.prefix[start1].name == fset.Sub:
            root_indexes = [i for i in range(len(subtree))
                            if subtree[i].name == fset.Sub or subtree[i].name == fset.G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            start2, end2 = _subtree_index(prefix=subtree, start_pos=rand_start_pos)
        else:
            start2 = random.randint(0, len(subtree) - 1)
            start2, end2 = _subtree_index(prefix=subtree, start_pos=start2)

        # gen new prefix
        ret = [self.prefix[i] for i in range(0, start1)]
        ret += [subtree[i] for i in range(start2, end2)]
        ret += [self.prefix[i] for i in range(end1, len(self))]

        self.prefix = ret
        self.get_depth_of_program()
