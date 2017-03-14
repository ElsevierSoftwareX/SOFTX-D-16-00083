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
from shutil import copyfile
import numpy as np
import sys
from fractions import Fraction as frac
import re
from XaNSoNSconfig import XaNSoNSconfig
def isstring(s):
    if (sys.version_info[0] == 3):
        return isinstance(s, str)
    return isinstance(s, basestring)
def str2list(X):
    Xnew=[]
    for x in X:
        Xnew.append(float(x.replace('(','').replace(')','')))
    return Xnew
def str2float(x):
    return float(x.replace('(','').replace(')',''))
def CheckFilename(filename):
    if not len(filename):
        return False
    pt=filename.rfind('.')
    fname=filename[0:pt]
    ext=filename[pt+1:]
    if ext.lower()!='cif':
        return False
    return True
def cif2xml(XML,backup='.back'):
    ciflag=False
    fromfile=False
    if isstring(XML):
        config=XaNSoNSconfig(XML)
        fromfile=True
    else:
        config=XML
    for Block in config.Block:
        ciflag=CheckFilename(Block.Atoms.filename)
    if not ciflag:
        return 0
    import CifFile as cif
    for Block in config.Block:
        if not CheckFilename(Block.Atoms.filename):
            continue
        cifile=cif.ReadCif(Block.Atoms.filename)
        keys=cifile.keys()
        for k in keys:
            cb=cifile[k]
            ids=[]
            occ=[]
            Uiso=[]
            if not cb.has_key('_atom_site_fract_x'):
                continue
            if cb.has_key('_atom_site_type_symbol'):
                ids=cb['_atom_site_type_symbol']
            elif cb.has_key('_atom_site_label'):
                ids_temp=cb['_atom_site_label']
                chars=set('0123456789()')
                for idsi in ids_temp:
                    pt=np.array([idsi.find(c) for c in chars])
                    pt=pt[np.where(pt>-1)]
                    ids.append(idsi[:pt.min()]) if pt.size else ids.append(idsi)
            X=cb['_atom_site_fract_x']
            X=str2list(X)
            Y=cb['_atom_site_fract_y']
            Y=str2list(Y)
            Z=cb['_atom_site_fract_z']
            Z=str2list(Z)
            if cb.has_key('_atom_site_occupancy'):
                occ=cb['_atom_site_occupancy']
                occ=str2list(occ)
            else:
                occ=[1.0 for id in ids]
            if Block.Atoms.disp:
                if cb.has_key('_atom_site_U_iso_or_equiv'):
                    Uiso=cb['_atom_site_U_iso_or_equiv']
                elif cb.has_key('_atom_site_U_equiv_geom_mean'):
                    Uiso=cb['_atom_site_U_equiv_geom_mean']
                elif cb.has_key('_atom_site_B_iso_or_equiv'):
                    Uiso=["%g"%(float(b)/(8*np.pi**2)) for b in cb['_atom_site_B_iso_or_equiv']]
                elif cb.has_key('_atom_site_B_equiv_geom_mean'):
                    Uiso=["%g"%(float(b)/(8*np.pi**2)) for b in cb['_atom_site_B_equiv_geom_mean']]
            if len(Uiso):
                Uiso=str2list(Uiso)
            else:
                Uiso=[0. for id in ids]
            a=str2float(cb['_cell_length_a'])
            b=str2float(cb['_cell_length_b'])
            c=str2float(cb['_cell_length_c'])
            alpha=str2float(cb['_cell_angle_alpha'])/180.*np.pi
            beta=str2float(cb['_cell_angle_beta'])/180.*np.pi
            gamma=str2float(cb['_cell_angle_gamma'])/180.*np.pi
            bc2 = b**2 + c**2 - 2*b*c*np.cos(alpha)
            h2 = b * np.cos(gamma)
            h3 = b * np.sin(gamma)
            h4 = c * np.cos(beta)
            h5 = ((h2 - h4)**2 + h3**2 + c**2 - h4**2 - bc2)/(2 * h3)
            h6 = np.sqrt(c**2 - h4**2 - h5**2)
            Block.CellVectors.a=np.array([a, 0., 0.])
            Block.CellVectors.b=np.array([h2, h3, 0.])
            Block.CellVectors.c=np.array([h4, h5, h6])
            symkey=''
            if cb.has_key('_symmetry_equiv_pos_as_xyz'):
                symkey='_symmetry_equiv_pos_as_xyz'
            elif cb.has_key('_space_group_symop_operation_xyz'):
                symkey='_space_group_symop_operation_xyz'
            if len(symkey):
                symmlist=cb[symkey]
                ax=['x','y','z']
                for symm in symmlist:
                    r=[0,0,0]
                    R=[[0,0,0],[0,0,0],[0,0,0]]
                    symm=symm.split(',')
                    for k in range(len(symm)):
                        symm[k]=symm[k].replace(' ','')
                        for j in range(len(ax)):
                            ind=symm[k].rfind(ax[j])
                            if ind<0:
                                continue
                            if ind>0: sign=symm[k][ind-1]
                            else: sign='+'
                            if sign=='+': R[k][j]=1.
                            else: R[k][j]=-1.
                        dig=re.search('\d', symm[k])
                        if not dig:
                            continue
                        ind=dig.start()
                        if ind>0: ind-=1
                        r[k] = float(frac(symm[k][ind:].rstrip('xyz+-')))                        
                    if r==[0,0,0] and R==[[1.,0,0],[0,1.,0],[0,0,1.]]:
                        continue
                    Block.AddSymmEqPos()
                    Block.SymmEqPos[-1].r=np.array(r)
                    Block.SymmEqPos[-1].R1=np.array(R[0])
                    Block.SymmEqPos[-1].R2=np.array(R[1])
                    Block.SymmEqPos[-1].R3=np.array(R[2])
            else:
                if cb.has_key('_symmetry_space_group_name_Hall') or cb.has_key('_symmetry_space_group_name_H-M'):
                    print('Unable to process %s: Symmetry group is specified but symmetry equivalent positions are not provided explicitly'%Block.Atoms.filename)
                    return -1
            break
        if len(ids)>Block.Atoms.MaxAtoms2XML:
            txtfname=Block.Atoms.filename[0:Block.Atoms.filename.rfind('.')]+'.txt'
            ftxt=open(txtfname,'w')
            ftxt.write('%d\n\n'%len(ids))
            for i in range(len(ids)):
                ftxt.write('%s %g %g %g %g %g\n'%(ids[i],X[i],Y[i],Z[i],Uiso[i],occ[i]))
            ftxt.close()
            Block.Atoms.filename=txtfname
        else:
            for i in range(len(ids)):
                Block.Atoms.AddAtom()
                Block.Atoms.Atom[-1].name=ids[i]
                Block.Atoms.Atom[-1].r=np.array([X[i],Y[i],Z[i]])
                Block.Atoms.Atom[-1].occ=occ[i]
                Block.Atoms.Atom[-1].Uiso=Uiso[i]
            Block.Atoms.filename=""
        Block.fractional=True
    if fromfile:
        if len(backup):
            copyfile(XML,XML+backup)
        config.ToXML(XML)
    return 0
if  __name__ ==  "__main__" :
    import sys
    if len(sys.argv)<2:
        print("Error: path to XML file is needed")
    else:
        cif2xml(sys.argv[1])
