class Node:
    left = None
    right = None
    data = None

    def __init__(self, data):
        self.data = data

    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.data,end=' -> ')
            self.inorder(node.right)

    def preorder(self, node):
        if node:
            print(node.data,end=' - ')
            self.preorder(node.left)
            self.preorder(node.right)

    def postorder(self, node):
        if node:
            self.postorder(node.left)
            self.postorder(node.right)
            print(node.data,end=' - ')


if __name__ == '__main__':
    node = Node('Refund = Yes')
    node.left = Node('No')
    node.right = Node('Yes')
    node.inorder(node)
    print()
