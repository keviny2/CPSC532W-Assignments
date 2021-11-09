## Possible Bugs:

- primitives.put()
    - don't know what the exact behavior of put should be for torch tensors. After replacing the element, should the tensor still be in the computational graph or not?