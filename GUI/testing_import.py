'''import importlib.util
import sys

name = "pp_central_elastic"
x = importlib.import_module(name)
print(x.kugeln)
'''

import importlib.util
import sys

module_path = "C:/Users/Jaist/Documents/GitHub/BA_DEM/GUI/examples/pp_central_elastic.py"
module_name = 'pp_central_elastic'

spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(module.kugeln)
