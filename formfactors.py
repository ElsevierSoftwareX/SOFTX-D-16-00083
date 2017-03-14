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
import periodictable as pt
from periodictable import cromermann
from shutil import copyfile
import numpy as np
import os
import sys
from XaNSoNSconfig import XaNSoNSconfig
def isstring(s):
    if (sys.version_info[0] == 3):
        return isinstance(s, str)
    return isinstance(s, basestring)
def formfactors(XML,backup='.back'):
    ID=[]
    fromfile=False
    if isstring(XML):
        config = XaNSoNSconfig(XML)
        fromfile=True
    else:
        config=XML
    FFset=False
    if os.path.isfile(config.Calculation.FFfilename):
        FFset=True
    for Block in config.Block:
        if len(Block.Atoms.filename):
            ftext=open(Block.Atoms.filename,'r')
            for line in ftext.readlines():
                if line[0]=="#":
                    continue
                l=line.split()
                if len(l)>1:
                    try:
                        name="%s"%pt.elements[int(l[0])]
                    except:
                        name=l[0]
                    ID.append(name)
            ftext.close()
        else:
            for atom in Block.Atoms.Atom:
                if len(atom.name):
                    ID.append(atom.name)
                else:
                    ID.append('%s'%pt.elements[int(atom.Z)])
    if not len(config.Calculation.FFfilename):
        config.Calculation.FFfilename=config.Calculation.name+'_FFtable.txt'
        if fromfile:
            if len(backup) and not os.path.isfile(XML+backup):
                copyfile(XML,XML+backup)
            config.ToXML(XML)
    ID=sorted(set(ID))
    ID_FF=[]
    open_mode='w'
    if FFset:
        open_mode='a+'
        ID_FF=np.genfromtxt(config.Calculation.FFfilename,usecols=(0,),dtype=str).tolist()
        if config.Calculation.source.lower()=='xray' and config.Calculation.scenario.lower()!='pdfonly':
            ID_FF.remove('q')
            line=''
            f=open(config.Calculation.FFfilename,'r')
            while line.find('q')<0:
                line=f.readline()
            f.close()
            line=line.strip('q')
            q=np.fromstring(line,dtype=float,sep=' ')
    f=open(config.Calculation.FFfilename,open_mode)
    if not FFset and config.Calculation.source.lower()=='xray' and config.Calculation.scenario.lower()!='pdfonly':
        f.write('q ')
        q=np.linspace(config.Calculation.q.min,config.Calculation.q.max,config.Calculation.q.N,endpoint=True)
        q.tofile(f,sep=' ',format='%e')
        f.write('\n\n')
    if config.Calculation.source.lower()=='xray' and config.Calculation.scenario.lower()=='pdfonly':
        q=config.Calculation.PDF.q
    for name in ID:
        if ID_FF.count(name):
            continue
        if config.Calculation.source.lower()=='xray':
            f.write('%s '%name)
            elname=name[0].upper()+name.rstrip('0').lower()[1:]
            try:
                ff=pt.elements.isotope(elname).xray.f0(q)
            except:
                try:
                    pt_sign=max(elname.find('+'),elname.find('-'))
                    if -1 < pt_sign < len(elname)-1:
                        elname=elname[:pt_sign]+elname[:pt_sign-1:-1]
                    ff=cromermann.fxrayatq(elname,q)
                except:
                    print('Error: %s is not found in database'%name)
                    f.close()
                    return -1
            if config.Calculation.scenario.lower()!='pdfonly':
                ff.tofile(f,sep=' ',format='%e')
                f.write('\n\n')
            else:
                f.write('%s %e\n\n'%(name,ff))
        elif config.Calculation.source.lower()=='neutron':
            try:
                elname=name[0].upper()+name.rstrip('0123456789+-').lower()[1:]
                f.write('%s %e\n\n'%(name,pt.elements.isotope(elname).neutron.b_c))
            except:
                print('Error: %s is not found in database'%name)
                f.close()
                return -1
    f.close()
    return 0
if  __name__ ==  "__main__" :
    import sys
    if len(sys.argv)<2:
        print("Error: path to XML file is needed")
    else:
        formfactors(sys.argv[1])
