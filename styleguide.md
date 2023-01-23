## Style Guide for Code Base

### General

* Always format files before pushing
    * In pycharm this can be mapped to ctrl+s at `File | Settings | Tools | Actions on Save`

### Import Statements

* Use explicit import statements whenever possible e.g. `from torch.utils.data import Dataset`
* Always optimize statements before pushing
    * In pycharm this can be mapped to ctrl+s at `File | Settings | Tools | Actions on Save`

### Strings

* Use single quotes in all occasions except:
    * Nested string formatting
    * Docstrings

### Signatures

* Always use type hinting
* All auxiliary arguments should be keyword arguments
  (If default value is undesired then use `foo(bar, *, baz)` to make baz keyword argument)
* When passing arguments to a callable, if there are many arguments, provide each arg on a new line for improved
  readability. i.e.

```
foo(bar,
    baz,
    qux,
    ...)
``` 

### Comments

* Include docstring for classes, functions and methods whenever not self-documenting
* Include comments whenever code block is not self-documenting