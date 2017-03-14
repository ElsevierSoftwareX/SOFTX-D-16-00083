#!/usr/bin/python
# -*- coding: utf-8 -*-
#Copyright (C) 2015, NRC "Kurchatov institute", http://www.nrcki.ru/e/engl.html, Moscow, Russia
#Author: Vladislav Neverov, vs-never@hotmail.com, neverov_vs@nrcki.ru
#
#This file is part of XaNSoNS.
#
#XaNSoNS is free software: you can redistribute it and / or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#XaNSoNS is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function
import os, shlex
from subprocess import Popen
from cif2xml import cif2xml
from formfactors import formfactors
from XaNSoNSconfig import XaNSoNSconfig as config
XaNSoNSDIR=os.path.dirname(__file__)
def run(XML,version="OMP",Nproc=4,Nthread=0,devID=-1,platID=-1,backup=True,CIFcheck=True,FFcheck=True):
    argstr=""
    if backup:
        if CIFcheck:
            cif2xml(opt.XML)
        if FFcheck:
            formfactors(opt.XML)
    else:
        if CIFcheck:
            cif2xml(opt.XML,"")
        if FFcheck:
            formfactors(opt.XML,"")
    if version=="single":
        argstr="'%s/XaNSoNS' %s"%(XaNSoNSDIR,XML)
    elif version=="OMP":
        argstr="'%s/XaNSoNS_OMP' %s"%(XaNSoNSDIR,XML)
        if Nthread:
            argstr+=" %d"%Nthread
    elif version=="MPI":
        argstr="mpiexec -n %d '%s/XaNSoNS_MPI' %s"%(Nproc,XaNSoNSDIR,XML)
    elif version=="MPI_OMP":
        argstr="mpiexec -n %d '%s/XaNSoNS_MPI_OMP' %s"%(Nproc,XaNSoNSDIR,XML)
        if Nthread:
            argstr+=" %d"%Nthread
        else:
            argstr="mpiexec -n %d '%s/XaNSoNS_MPI_OMP' %s"%(Nproc,XaNSoNSDIR,XML)
    elif version=="CUDA":
        argstr="'%s/XaNSoNS_CUDA' %s %d"%(XaNSoNSDIR,XML,devID)
    elif version=="OpenCL":
        argstr="'%s/XaNSoNS_OCL' %s %d %d"%(XaNSoNSDIR,XML,devID,platID)
    else:
        print("Error: wrong version name: %s"%version)
        return -1
    if len(argstr):
        return Popen(shlex.split(argstr)) 
if  __name__ ==  "__main__" :
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--XML", type='string',dest="XML", 
                      default="start.xml",help="input XML file")
    parser.add_option("-v", "--version", dest="version", type='string', default="OMP",
                      help="Code version to execute (select from the list: single, MPI, OMP, MPI_OMP, CUDA, OpenCL)")
    parser.add_option("-N", "--Nproc", dest="Nproc", type='int',default=4,
                      help="Number of MPI processes")
    parser.add_option("-n", "--Nthread", dest="Nthread", type='int',default=0,
                      help="Number of OpenMP threads (0 - system defaults)")
    parser.add_option("-d", "--deviceID", dest="devID", type='int',default=-1,
                      help="CUDA or OpenCL device ID")
    parser.add_option("-p", "--platformID", dest="platID", type='int',default=-1,
                      help="OpenCL platform ID")
    parser.add_option("--nobackup", dest="backup", action='store_false',default=True,
                      help="Don't backup XMLs")
    parser.add_option("--noFFcheck", dest="FFcheck", action='store_false',default=True,
                      help="Don't check if the form factors are exist and valid")
    parser.add_option("--noCIFcheck", dest="CIFcheck", action='store_false',default=True,
                      help="Don't check for *.cif files")
    (opt, args) = parser.parse_args()
    run(opt.XML,version=opt.version,Nproc=opt.Nproc,Nthread=opt.Nthread,devID=opt.devID,platID=opt.platID,backup=opt.backup,FFcheck=opt.FFcheck,CIFcheck=opt.CIFcheck).wait()