import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/install/object_detection'
