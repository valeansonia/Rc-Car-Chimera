import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/arrk-adas/Desktop/Rc-Car-Chimera/install/acc_drive'
