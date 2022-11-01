import random

from .fset import *

# #################################################################################
# Multi-Layer GP tree structure.
# #################################################################################
# Classification Layer =======> class_0 <= [result <= 0] | [result > 0] => class_1
#                                                        |
#                                                      [Sub]
#                                                     /     \
# Feature Construction Layer =>                 [G_Std]      [G_Std]
#                                                  |            |
#                                             [Hist_Eq]      [Lap]
#                                                  |            |
# Feature Extraction Layer ===>               [Sobel_X]      [Sobel_Y]
#                                                  |            |
# Region Detection Layer =====>              [Region_S]      [Region_R]
#                                                  |            |
# Input Layer ================>        [rx, ry, rh, rw]      [rx, ry, rh, rw]
#
# #################################################################################


terminal_set = [Region_S, Region_R]
inter_func_set = [Hist_Eq, Gau1, Lap, Sobel_X, Sobel_Y, LoG1]


def _rand_term() -> int:
    """Sample a function from Region_S and Region_R."""
    return random.sample(population=terminal_set, k=1)[0]


def _rand_sub_gstd(sub_rate=0.3) -> int:
    """Sample a function from Sub and G_Std.
    Args:
        sub_rate: the probability to generate Sub function.
    """
    rand_float = random.random()
    return Sub if rand_float < sub_rate else G_Std


def _rand_inter_func() -> int:
    """Sample a function from feature extraction functions, exclude G_Std."""
    return random.sample(population=inter_func_set, k=1)[0]


class Node:
    def __init__(self, name: int, rx=0, ry=0, rh=0, rw=0):
        self.name = name
        self.rx = rx
        self.ry = ry
        self.rh = rh
        self.rw = rw

    def __repr__(self):
        return to_str(self.name)

    def __str__(self):
        return to_str(self.name) + '(' + str(self.rx) + ', ' + str(self.ry) + ', ' \
               + str(self.rh) + ', ' + str(self.rw) + ')' if self.is_terminal() else to_str(self.name)

    def is_binary_function(self):
        return is_binary_function(self.name)

    def is_unary_function(self):
        return is_unary_function(self.name)

    def is_terminal(self):
        return is_terminal(self.name)


def rand_terminal_node(img_h, img_w) -> Node:
    """Create a random terminal node with respect to the height and width of the image."""
    if img_h < 20 or img_w < 20:
        raise RuntimeError('The height and width must be larger or equal to 20.')

    reg_x = random.randint(0, img_h - 20)
    reg_y = random.randint(0, img_w - 20)

    if _rand_term() == Region_S:
        max_side_len = min(img_h - reg_x, img_w - reg_y)
        side_len = random.randint(20, max_side_len)
        node = Node(name=Region_S, rx=reg_x, ry=reg_y, rh=side_len, rw=side_len)
        return node

    else:
        reg_h = random.randint(20, img_h - reg_x)
        reg_w = random.randint(20, img_w - reg_y)
        node = Node(name=Region_R, rx=reg_x, ry=reg_y, rh=reg_h, rw=reg_w)
        return node


def rand_inter_func_node() -> Node:
    """Create a random function node from feature extraction functions exclude G_Std."""
    name = _rand_inter_func()
    return Node(name=name)


class TreeGenerator:

    class TreeNode:
        def __init__(self, node=None, left=None, right=None):
            self.node = node
            self.left = left
            self.right = right

    def __init__(self, depth, img_h, img_w):
        self.depth = depth
        self.img_h = img_h
        self.img_w = img_w

    def full_init_tree(self) -> List[Node]:
        root_tree_node = self._create_full_tree(self.depth)
        ret = []
        self._get_init_prefix(prefix=ret, tree_node=root_tree_node)
        return ret

    def growth_init_tree(self) -> List[Node]:
        root_tree_node = self._create_growth_tree(self.depth)
        ret = []
        self._get_init_prefix(prefix=ret, tree_node=root_tree_node)
        return ret

    def _get_init_prefix(self, prefix: List[Node], tree_node):
        if tree_node is None:
            return
        prefix.append(tree_node.node)
        self._get_init_prefix(prefix, tree_node.left)
        self._get_init_prefix(prefix, tree_node.right)

    def _create_full_tree(self, depth, parent_node=None) -> TreeNode:
        if parent_node is None:
            tree_node = self.TreeNode(node=Node(Sub))
            tree_node.left = self._create_full_tree(depth - 1, Sub)
            tree_node.right = self._create_full_tree(depth - 1, Sub)
            return tree_node

        if depth == 2 and parent_node == Sub:
            node = Node(name=G_Std)
            tree_node = self.TreeNode(node=node)
            tree_node.left = self._create_full_tree(depth - 1, G_Std)
            return tree_node

        if depth == 1:
            terminal = rand_terminal_node(img_h=self.img_h, img_w=self.img_w)
            tree_node = self.TreeNode(node=terminal)
            return tree_node

        if parent_node is Sub:
            func = _rand_sub_gstd()
            node = Node(name=func)
            tree_node = self.TreeNode(node=node)
            if func == G_Std:
                tree_node.left = self._create_full_tree(depth - 1, G_Std)
                return tree_node

            else:
                tree_node.left = self._create_full_tree(depth - 1, Sub)
                tree_node.right = self._create_full_tree(depth - 1, Sub)
                return tree_node

        else:
            node = rand_inter_func_node()
            tree_node = self.TreeNode(node=node)
            tree_node.left = self._create_full_tree(depth - 1, node.name)
            return tree_node

    def _create_growth_tree(self, depth, parent_node=None, return_rate=0.5):
        if parent_node is None:
            tree_node = self.TreeNode(node=Node(Sub))
            tree_node.left = self._create_growth_tree(depth - 1, Sub)
            tree_node.right = self._create_growth_tree(depth - 1, Sub)
            return tree_node

        if depth == 2 and parent_node == Sub:
            node = Node(name=G_Std)  # must be S_Std, can not be Sub
            tree_node = self.TreeNode(node=node)
            tree_node.left = self._create_growth_tree(depth - 1, G_Std)
            return tree_node

        if depth == 1:
            terminal = rand_terminal_node(img_h=self.img_h, img_w=self.img_w)
            tree_node = self.TreeNode(node=terminal)
            return tree_node

        if parent_node is Sub:
            func = _rand_sub_gstd()
            node = Node(name=func)
            tree_node = self.TreeNode(node=node)
            if func == G_Std:
                tree_node.left = self._create_growth_tree(depth - 1, G_Std)
                return tree_node

            else:
                tree_node.left = self._create_growth_tree(depth - 1, Sub)
                tree_node.right = self._create_growth_tree(depth - 1, Sub)
                return tree_node

        else:
            if return_rate is None:
                func_set = terminal_set + inter_func_set
                rand_func = random.sample(population=func_set, k=1)[0]
                node = Node(name=rand_func)
            else:
                if random.random() < return_rate:
                    node = rand_terminal_node(self.img_h, self.img_w)
                else:
                    node = rand_inter_func_node()

            tree_node = self.TreeNode(node=node)

            if node.is_terminal():
                return tree_node

            tree_node.left = self._create_growth_tree(depth - 1, node.name)
            return tree_node
