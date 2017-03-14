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
import inspect
import numpy as np
from xml.dom import minidom
import codecs
def string2bool(string):
    strl=string.lower()
    if strl == 'yes' or strl == 'true' or strl == '1':
        return True
    return False
def bool2string(flag):
    if flag==True:
        return "yes"
    return "no"
class XMLparamBase:
    def GetFromXMLelement(self,XMLelement):
        for member in inspect.getmembers(self, lambda a:not(inspect.isroutine(a))):
            if member[0][:2]=='__':
                continue
            attr=getattr(self,member[0])
            membertype=type(attr)
            if XMLelement.hasAttribute(member[0]):
                if membertype==float:
                    setattr(self,member[0],float(XMLelement.getAttribute(member[0])))
                elif membertype==int:
                    setattr(self,member[0],int(XMLelement.getAttribute(member[0])))
                elif membertype==bool:
                    setattr(self,member[0],string2bool(XMLelement.getAttribute(member[0])))
                elif membertype==str:
                    setattr(self,member[0],XMLelement.getAttribute(member[0]))
                elif membertype==np.ndarray:
                    tempstr=XMLelement.getAttribute(member[0])
                    if attr.dtype==np.int32:
                        setattr(self,member[0],np.array(tempstr.split()).astype(np.int32))
                    else:
                        setattr(self,member[0],np.array(tempstr.split()).astype(np.float64))
            else:
                ChildElements=XMLelement.getElementsByTagName(member[0])
                if not len(ChildElements):
                    continue
                if membertype==list:
                    tempMethod=getattr(self,"Add"+member[0])
                    for el in ChildElements:
                        tempMethod()
                        atr=getattr(self,member[0])[-1]
                        atr.GetFromXMLelement(el)
                else:
                    atr=getattr(self,member[0])
                    atr.GetFromXMLelement(ChildElements[0])
    def Show(self,prefix=""):
        for member in inspect.getmembers(self, lambda a:not(inspect.isroutine(a))):
            if member[0][:2]=='__':
                continue
            tempstr=prefix+self.__name__+"-->"+member[0]+" = "
            attr=getattr(self,member[0])
            membertype=type(attr)
            if membertype==np.ndarray:
                print(tempstr+" ".join(["%g"%i for i in attr]))
            elif membertype==bool:
                print(tempstr+bool2string(attr))
            elif membertype==float or membertype==int:
                print(tempstr+"%g"%attr)
            elif membertype==str or membertype==unicode:
                print(tempstr+attr)
            elif membertype==list:
                for i in range(len(attr)):
                    attr[i].Show(prefix+self.__name__+"-->%d:"%i)
            else:
                attr.Show(prefix+self.__name__+"-->")
class qParam(XMLparamBase):
    def __init__(self):
        self.__name__="q"
        self.min=-1.
        self.max=-1.
        self.N=1024
        self.Nfi=1024
class EulerParam(XMLparamBase):
    def __init__(self):
        self.__name__="Euler"
        self.min=np.zeros(3,dtype=np.float64)
        self.max=np.zeros(3,dtype=np.float64)
        self.N=np.ones(3,dtype=np.int32)
        self.convention="ZXZ"
    def ToXML(self,doc,PrintDefaults=False):
        EulerEl=doc.createElement("Euler")
        createEuler=False
        defaultEuler=EulerParam()
        if  (defaultEuler.min!=self.min).any() or PrintDefaults:
            EulerEl.setAttributeNode(doc.createAttribute("min"))
            EulerEl.setAttribute("min"," ".join(["%g"%i for i in self.min]))
            createEuler=True
        if  (defaultEuler.max!=self.max).any() or PrintDefaults:
            EulerEl.setAttributeNode(doc.createAttribute("max"))
            EulerEl.setAttribute("max"," ".join(["%g"%i for i in self.max]))
            createEuler=True
        if  (defaultEuler.N!=self.N).any() or PrintDefaults:
            EulerEl.setAttributeNode(doc.createAttribute("N"))
            EulerEl.setAttribute("N"," ".join(["%g"%i for i in self.N]))
            createEuler=True
        if  defaultEuler.convention!=self.convention or PrintDefaults:
            EulerEl.setAttributeNode(doc.createAttribute("convention"))
            EulerEl.setAttribute("convention",self.convention)
            createEuler=True
        if not createEuler:
            return -1
        return EulerEl
class PDFParam(XMLparamBase):
    def __init__(self):
        self.__name__="PDF"
        self.type="PDF"
        self.q=0.
        self.density=0.
        self.PrintPartial=False
    def ToXML(self,doc,PrintDefaults=False):
        PDFEl=doc.createElement("PDF")
        defaultPDF=PDFParam()
        createPDF=False
        if  defaultPDF.type!=self.type or PrintDefaults:
            PDFEl.setAttributeNode(doc.createAttribute("type"))
            PDFEl.setAttribute("type",self.type)
            createPDF=True
        if  defaultPDF.q!=self.q or PrintDefaults:
            PDFEl.setAttributeNode(doc.createAttribute("q"))
            PDFEl.setAttribute("q","%g"%self.q)
            createPDF=True
        if  defaultPDF.density!=self.density or PrintDefaults:
            PDFEl.setAttributeNode(doc.createAttribute("density"))
            PDFEl.setAttribute("density","%g"%self.density)
            createPDF=True
        if  defaultPDF.PrintPartial!=self.PrintPartial or PrintDefaults:
            PDFEl.setAttributeNode(doc.createAttribute("PrintPartial"))
            PDFEl.setAttribute("PrintPartial",bool2string(self.PrintPartial))
            createPDF=True
        if not createPDF:
            return -1
        return PDFEl
class CalculationParam(XMLparamBase):
    def __init__(self):
        self.__name__="Calculation"
        self.scenario = "debye"
        self.source = "xray"
        self.name = ""
        self.PrintAtoms=False
        self.FFfilename=""
        self.PolarFactor=False
        self.q=qParam()
        self.wavelength=-1.
        self.Euler=EulerParam()
        self.PartialIntensity=False
        self.hist_bin=0.001
        self.PDF=PDFParam()
    def ToXML(self,doc,PrintDefaults=False):
        defaultCalc=CalculationParam()
        CalcEl=doc.createElement("Calculation")
        if len(self.name):
            CalcEl.setAttributeNode(doc.createAttribute("name"))
            CalcEl.setAttribute("name",self.name)
        else:
            print("XML printing error: Calculation-->name is not set")
            return -1
        if  defaultCalc.source!=self.source or PrintDefaults:
            CalcEl.setAttributeNode(doc.createAttribute("source"))
            CalcEl.setAttribute("source",self.source)
        if  defaultCalc.scenario!=self.scenario or PrintDefaults:
            CalcEl.setAttributeNode(doc.createAttribute("scenario"))
            CalcEl.setAttribute("scenario",self.scenario)
        if  defaultCalc.PrintAtoms!=self.PrintAtoms or PrintDefaults:
            CalcEl.setAttributeNode(doc.createAttribute("PrintAtoms"))
            CalcEl.setAttribute("PrintAtoms",bool2string(self.PrintAtoms))
        if  defaultCalc.FFfilename!=self.FFfilename or PrintDefaults:
            CalcEl.setAttributeNode(doc.createAttribute("FFfilename"))
            CalcEl.setAttribute("FFfilename",self.FFfilename)
        if self.scenario.lower()!="pdfonly":
            if self.source.lower()=="xray" and (defaultCalc.PolarFactor!=self.PolarFactor or PrintDefaults):
                CalcEl.setAttributeNode(doc.createAttribute("PolarFactor"))
                CalcEl.setAttribute("PolarFactor",bool2string(self.PolarFactor))
            if defaultCalc.wavelength!=self.wavelength:
                CalcEl.setAttributeNode(doc.createAttribute("wavelength"))
                CalcEl.setAttribute("wavelength","%g"%self.wavelength)
            elif self.PolarFactor:
                print("XML printing error: Calculation-->wavelength is not set")
                return -1
            qEl=doc.createElement("q")
            if self.q.min!=defaultCalc.q.min:
                qEl.setAttributeNode(doc.createAttribute("min"))
                qEl.setAttribute("min","%g"%self.q.min)
            else:
                print("XML printing error: Calculation-->q-->min is not set")
            if self.q.max!=defaultCalc.q.max:
                qEl.setAttributeNode(doc.createAttribute("max"))
                qEl.setAttribute("max","%g"%self.q.max)
            else:
                print("XML printing error: Calculation-->q-->max is not set")
            if self.q.N!=defaultCalc.q.N or PrintDefaults:
                qEl.setAttributeNode(doc.createAttribute("N"))
                qEl.setAttribute("N","%g"%self.q.N)
            if self.scenario.lower()=="2d":
                if self.q.Nfi!=defaultCalc.q.Nfi or PrintDefaults:
                    qEl.setAttributeNode(doc.createAttribute("Nfi"))
                    qEl.setAttribute("Nfi","%g"%self.q.Nfi)
                EulerEl=self.Euler.ToXML(doc,PrintDefaults)
                if EulerEl!=-1:
                    CalcEl.appendChild(EulerEl)
            CalcEl.appendChild(qEl)
            if self.scenario.lower()=="debye":
                if  defaultCalc.PartialIntensity!=self.PartialIntensity or PrintDefaults:
                    CalcEl.setAttributeNode(doc.createAttribute("PartialIntensity"))
                    CalcEl.setAttribute("PartialIntensity",bool2string(self.PartialIntensity))
        if self.scenario.lower()=="debye_hist" or self.scenario.lower()=="pdfonly" or self.scenario.lower()=="debyepdf":
            if  defaultCalc.hist_bin!=self.hist_bin or PrintDefaults:
                CalcEl.setAttributeNode(doc.createAttribute("hist_bin"))
                CalcEl.setAttribute("hist_bin","%g"%self.hist_bin)
        if self.scenario.lower()=="pdfonly" or self.scenario.lower()=="debyepdf":
            PDFEl=self.PDF.ToXML(doc,PrintDefaults)
            if PDFEl!=-1:
                CalcEl.appendChild(PDFEl)
        return CalcEl
class AtomParam(XMLparamBase):
    def __init__(self):
        self.__name__="Atom"
        self.name=""
        self.Z=0
        self.r=np.zeros(3,dtype=np.float64)
        self.occ=1.0
        self.Uiso=0.
    def ToXML(self,doc,PrintDefaults=False):
        defaultAtom=AtomParam()
        AtomEl=doc.createElement("Atom")
        if not self.Z+len(self.name):
            print("XML printing error: Neither Z nor name of the atom are set")
            return -1
        if len(self.name):
            AtomEl.setAttributeNode(doc.createAttribute("name"))
            AtomEl.setAttribute("name",self.name)
        else:
            AtomEl.setAttributeNode(doc.createAttribute("Z"))
            AtomEl.setAttribute("Z","%g"%self.Z)
        AtomEl.setAttributeNode(doc.createAttribute("r"))
        AtomEl.setAttribute("r"," ".join(["%g"%i for i in self.r]))
        if  defaultAtom.occ!=self.occ or PrintDefaults:
            AtomEl.setAttributeNode(doc.createAttribute("occ"))
            AtomEl.setAttribute("occ","%g"%self.occ)
        if  defaultAtom.Uiso!=self.Uiso or PrintDefaults:
            AtomEl.setAttributeNode(doc.createAttribute("Uiso"))
            AtomEl.setAttribute("Uiso","%g"%self.Uiso)
        return AtomEl
class AtomsParam(XMLparamBase):
    def __init__(self):
        self.__name__="Atoms"
        self.filename=""
        self.Atom=[]
        self.disp=True
        self.MaxAtoms2XML=50
    def AddAtom(self):
        self.Atom.append(AtomParam())
    def ToXML(self,doc,PrintDefaults=False):
        if not (len(self.Atom) + len(self.filename)):
            print("XML printing error: Atomic data are not set")
            return -1
        defaultAtoms=AtomsParam()
        AtomsEl=doc.createElement("Atoms")
        if  len(self.filename):
            AtomsEl.setAttributeNode(doc.createAttribute("filename"))
            AtomsEl.setAttribute("filename",self.filename)
            if self.filename[-4:].lower()==".cif":
                if  defaultAtoms.disp!=self.disp or PrintDefaults:
                    AtomsEl.setAttributeNode(doc.createAttribute("disp"))
                    AtomsEl.setAttribute("disp",bool2string(self.disp))
                if  defaultAtoms.MaxAtoms2XML!=self.MaxAtoms2XML or PrintDefaults:
                    AtomsEl.setAttributeNode(doc.createAttribute("MaxAtoms2XML"))
                    AtomsEl.setAttribute("MaxAtoms2XML","%g"%self.MaxAtoms2XML)
        else:
            for atom in self.Atom:
                AtomEl=atom.ToXML(doc,PrintDefaults)
                if AtomEl==-1:
                    return -1
                AtomsEl.appendChild(AtomEl)
        return AtomsEl
class SymmEqPosParam(XMLparamBase):
    def __init__(self):
        self.__name__="SymmEqPos"
        self.r=np.zeros(3,dtype=np.float64)
        self.R1=np.zeros(3,dtype=np.float64)
        self.R2=np.zeros(3,dtype=np.float64)
        self.R3=np.zeros(3,dtype=np.float64)
    def ToXML(self,doc):
        symmEl=doc.createElement("SymmEqPos")
        symmEl.setAttributeNode(doc.createAttribute("r"))
        symmEl.setAttribute("r"," ".join(["%g"%i for i in self.r]))
        symmEl.setAttributeNode(doc.createAttribute("R1"))
        symmEl.setAttribute("R1"," ".join(["%g"%i for i in self.R1]))
        symmEl.setAttributeNode(doc.createAttribute("R2"))
        symmEl.setAttribute("R2"," ".join(["%g"%i for i in self.R2]))
        symmEl.setAttributeNode(doc.createAttribute("R3"))
        symmEl.setAttribute("R3"," ".join(["%g"%i for i in self.R3]))
        return symmEl
class CutOffParam(XMLparamBase):
    def __init__(self):
        self.__name__="CutOff"
        self.Rcut=0.
        self.RcutCopies=0.
class RearrangementParam(XMLparamBase):
    def __init__(self):
        self.__name__="Rearrangement"
        self.filename=""
        self.Rcorr=0.
class ClusteringParam(XMLparamBase):
    def __init__(self):
        self.__name__="Clustering"
        self.N=0
        self.Rmin=0.
class CellVectorsParam(XMLparamBase):
    def __init__(self):
        self.__name__="CellVectors"
        self.a=np.array([1,0,0],dtype=np.float64)
        self.b=np.array([0,1,0],dtype=np.float64)
        self.c=np.array([0,0,1],dtype=np.float64)
        self.N=np.ones(3,dtype=np.int64)
    def ToXML(self,doc,PrintDefaults=False):
        defaultCell=CellVectorsParam()
        CellEl=doc.createElement("CellVectors")
        addCell=False
        if (defaultCell.a!=self.a).any() or PrintDefaults:
            CellEl.setAttributeNode(doc.createAttribute("a"))
            CellEl.setAttribute("a"," ".join(["%g"%i for i in self.a]))
            addCell=True
        if (defaultCell.b!=self.b).any() or PrintDefaults:
            CellEl.setAttributeNode(doc.createAttribute("b"))
            CellEl.setAttribute("b"," ".join(["%g"%i for i in self.b]))
            addCell=True
        if (defaultCell.c!=self.c).any() or PrintDefaults:
            CellEl.setAttributeNode(doc.createAttribute("c"))
            CellEl.setAttribute("c"," ".join(["%g"%i for i in self.c]))
            addCell=True
        if (defaultCell.N!=self.N).any() or PrintDefaults:
            CellEl.setAttributeNode(doc.createAttribute("N"))
            CellEl.setAttribute("N"," ".join(["%g"%i for i in self.N]))
            addCell=True
        if not addCell:
            return -1
        return CellEl
class CopyParam(XMLparamBase):
    def __init__(self):
        self.__name__="Copy"
        self.r=np.zeros(3,dtype=np.float64)
        self.Euler=np.zeros(3,dtype=np.float64)
    def ToXML(self,doc):
        CopyEl=doc.createElement("Copy")
        CopyEl.setAttributeNode(doc.createAttribute("r"))
        CopyEl.setAttribute("r"," ".join(["%g"%i for i in self.r]))
        CopyEl.setAttributeNode(doc.createAttribute("Euler"))
        CopyEl.setAttribute("Euler"," ".join(["%g"%i for i in self.Euler]))
        return CopyEl
class CopiesParam(XMLparamBase):
    def __init__(self):
        self.__name__="Copies"
        self.filename=""
        self.convention="ZYX"
        self.Clustering=ClusteringParam()
        self.Copy=[]
    def AddCopy(self):
        self.Copy.append(CopyParam())
    def ToXML(self,doc,PrintDefaults=False):
        defaultCopies=CopiesParam()
        CopiesEl=doc.createElement("Copies")
        if not (len(self.Copy) + len(self.filename) + self.Clustering.N):
            return -1
        if  defaultCopies.convention!=self.convention or PrintDefaults:
            CopiesEl.setAttributeNode(doc.createAttribute("convention"))
            CopiesEl.setAttribute("convention",self.convention)
        if  self.Clustering.N and self.Clustering.Rmin:
            ClustEl=doc.createElement("Clustering")
            ClustEl.setAttributeNode(doc.createAttribute("N"))
            ClustEl.setAttribute("N","%g"%self.Clustering.N)
            ClustEl.setAttributeNode(doc.createAttribute("Rmin"))
            ClustEl.setAttribute("Rmin","%g"%self.Clustering.Rmin)
            CopiesEl.appendChild(ClustEl)
        if  len(self.filename):
            CopiesEl.setAttributeNode(doc.createAttribute("filename"))
            CopiesEl.setAttribute("filename",self.filename)
        else:
            for copy in self.Copy:
                CopyEl=copy.ToXML(doc)
                CopiesEl.appendChild(CopyEl)
        return CopiesEl
class BlockParam(XMLparamBase):
    def __init__(self):
        self.__name__="Block"
        self.fractional=False
        self.centered=False
        self.centeredAtoms=False
        self.mol_rotation=False
        self.mol_Uiso=0.
        self.Atoms=AtomsParam()
        self.SymmEqPos=[]
        self.CutOff=CutOffParam()
        self.Copies=CopiesParam()
        self.Rearrangement=RearrangementParam()
        self.CellVectors=CellVectorsParam()
    def AddSymmEqPos(self):
        self.SymmEqPos.append(SymmEqPosParam())
    def ToXML(self,doc,PrintDefaults=False):
        defaultBlock=BlockParam()
        blockEl=doc.createElement("Block")
        if  defaultBlock.fractional!=self.fractional or PrintDefaults:
            blockEl.setAttributeNode(doc.createAttribute("fractional"))
            blockEl.setAttribute("fractional",bool2string(self.fractional))
        if  defaultBlock.centered!=self.centered or PrintDefaults:
            blockEl.setAttributeNode(doc.createAttribute("centered"))
            blockEl.setAttribute("centered",bool2string(self.centered))
        if  defaultBlock.centeredAtoms!=self.centeredAtoms or PrintDefaults:
            blockEl.setAttributeNode(doc.createAttribute("centeredAtoms"))
            blockEl.setAttribute("centeredAtoms",bool2string(self.centeredAtoms))
        if  defaultBlock.mol_rotation!=self.mol_rotation or PrintDefaults:
            blockEl.setAttributeNode(doc.createAttribute("mol_rotation"))
            blockEl.setAttribute("mol_rotation",bool2string(self.mol_rotation))
        if  defaultBlock.mol_Uiso!=self.mol_Uiso or PrintDefaults:
            blockEl.setAttributeNode(doc.createAttribute("mol_Uiso"))
            blockEl.setAttribute("mol_Uiso","%g"%self.mol_Uiso)
        AtomsEl=self.Atoms.ToXML(doc,PrintDefaults)
        if AtomsEl==-1:
            return -1   
        blockEl.appendChild(AtomsEl)
        for symm in self.SymmEqPos:
            symmEl=symm.ToXML(doc)
            blockEl.appendChild(symmEl)
        CellEl=self.CellVectors.ToXML(doc,PrintDefaults)
        if CellEl!=-1:
            blockEl.appendChild(CellEl)
        CopiesEl=self.Copies.ToXML(doc,PrintDefaults)
        if CopiesEl!=-1:
            blockEl.appendChild(CopiesEl)
        if self.CutOff.Rcut or self.CutOff.RcutCopies:
            CutEl=doc.createElement("CutOff")
            if self.CutOff.Rcut:
                CutEl.setAttributeNode(doc.createAttribute("Rcut"))
                CutEl.setAttribute("Rcut","%g"%self.CutOff.Rcut)
            if self.CutOff.RcutCopies:
                CutEl.setAttributeNode(doc.createAttribute("RcutCopies"))
                CutEl.setAttribute("RcutCopies","%g"%self.CutOff.RcutCopies)
            blockEl.appendChild(CutEl)
        if len(self.Rearrangement.filename):
            RearrEl=doc.createElement("Rearrangement")
            RearrEl.setAttributeNode(doc.createAttribute("filename"))
            RearrEl.setAttribute("filename",self.Rearrangement.filename)
            RearrEl.setAttributeNode(doc.createAttribute("Rcorr"))
            RearrEl.setAttribute("Rcorr","%g"%self.Rearrangement.Rcorr)
            blockEl.appendChild(RearrEl)
        return blockEl
class XaNSoNSconfig(XMLparamBase):
    def __init__(self,XMLfile=None):
        self.__name__="Config"
        self.Calculation=CalculationParam()
        self.Block=[]
        if XMLfile is not None:
            self.FromXML(XMLfile)
    def AddBlock(self):
        self.Block.append(BlockParam())
    def Print(self):
        self.Calculation.Show()
        for i in range(len(self.Block)):
            self.Block[i].Show("%d:"%i)
    def FromXML(self,XMLfile):
        doc = minidom.parse(XMLfile)
        self.GetFromXMLelement(doc.firstChild)
    def ToXML(self,XMLfile,PrintDefaults=False):
        doc = minidom.Document()
        mainEl=doc.createElement("PMML")
        CalcEl=self.Calculation.ToXML(doc,PrintDefaults)
        if CalcEl==-1:
            return -1       
        mainEl.appendChild(CalcEl)
        for block in self.Block:
            blockEl=block.ToXML(doc,PrintDefaults)
            if blockEl==-1:
                return blockEl
            mainEl.appendChild(blockEl)
        doc.appendChild(mainEl)
        with codecs.open(XMLfile, 'w', encoding='utf8') as out:
            out.write(doc.toprettyxml(newl='\r'))
 
