#!/bin/bash

#Test if the environment variables EDISCS and EAZYINSTALL exist. If
#not, exit. If so, copy files and make a symbolic link to the EAZY
#spectroscopic templates.

if [ -z ${EDISCS+'tmp'} ]; then
    echo "Set system variable EDISCS!" ;
    exit ;
else
    cp ${EDISCS}/files/analysis/EAZY/filters.dat .
    cp ${EDISCS}/files/analysis/EAZY/zphot.param .
fi

if [ -z ${EAZYINSTALL+'tmp'} ]; then
    echo "Set system variable EAZYINSTALL!" ;
    exit ;
else
    ln -s ${EAZYINSTALL}/templates templates ;
fi

#If the directory OUTPUT does not exist, then make it.
if [ ! -d "OUTPUT" ]; then
   mkdir OUTPUT;
fi


