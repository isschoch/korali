def getcflags(extdir, makeFlags):
  fixedflags = '-g -O3 -Wno-deprecated-declarations -I' + extdir
  print(fixedflags + ' ' + makeFlags['OPTFLAGS'] + ' ' + makeFlags['MPIFLAGS'] +
        ' ' + makeFlags['GSLCFLAGS'] + ' ' + makeFlags['RTNORMCFLAGS'] + ' ' +
        makeFlags['LIBCOCFLAGS'] + ' ' + makeFlags['EIGENCFLAGS'] + ' ' +
        makeFlags['ONEDNNCFLAGS'] + ' ' + makeFlags['CXXARCH'] + ' ' +
        makeFlags['PYBIND11INCLUDES'])
