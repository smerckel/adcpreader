#!/usr/bin/python3

import argparse

from datetime import datetime, timezone
import glob
import os
import sys
import shutil
import numpy as np

import dbdreader
import rdi

parser = argparse.ArgumentParser(description='Rename PD0 files in dbdreader dbd files.')


parser.add_argument("glider", help = "glider name")
parser.add_argument("dbd_directory", help = "path to directory where the dbd files are found.")
parser.add_argument("pd0_directory", help = "path to directory where the pd0 files are found.")

parser.add_argument("--dbd_ext", choices="dbd ebd sbd tbd DBD EBD SBD TBD".split(), help = "extension of the dbd files.", default = "dbd")
parser.add_argument("--pd0_ext", choices="pd0 PD0".split(), help = "extension of the pd0 files.", default = "pd0")

#parser.add_argument("-o","--operation", choices="copy move symlink".split(), help = "Operation to be carried out on the original file(s).", default = "copy")


args = parser.parse_args()

pd0_ext = args.pd0_ext
dbd_ext = args.dbd_ext
#operation = args.operation
operation = "copy"

for s in [args.dbd_directory, args.pd0_directory]:
    if not os.path.exists(s):
        raise IOError('"{}" does not exist.'.format(s))
    if not os.path.isdir(s):
        raise IOError('"{}" is not an directory.'.format(s))

fns = dbdreader.DBDList(glob.glob(os.path.join(args.dbd_directory,"{}*.{}".format(args.glider, dbd_ext))))
if not fns:
    raise ValueError("No {} files were found in {}.".format(args.dbd_ext, args.dbd_directory))
fns.sort()


pd0s = glob.glob(os.path.join(args.pd0_directory,"*.{}".format(pd0_ext)))
if not pd0s:
    raise ValueError("No {} files were found in {}.".format(args.pd0_ext, args.pd0_directory))
pd0s.sort()

t_pd0 = []
f_pd0 = []

pd0 = rdi.rdi_reader.PD0()
for j,fn in enumerate(pd0s):
    for i,ens in enumerate(pd0.ensemble_generator((fn,))):
        rtc = list(ens['variable_leader']['RTC'])
        rtc[0]+=2000
        rtc[6]*=1000
        tm = datetime(*rtc, timezone.utc).timestamp()
        break
    t_pd0.append(tm)
    f_pd0.append(fn)

sys.stdout.write("Found %d PD0 files.\n"%(len(t_pd0)))

n_processed = 0
for fn in fns:
    dbd = dbdreader.DBD(fn)
    t,_ = dbd.get("m_depth")
    dbd.close()
    try:
        t0 = t[0]
    except IndexError:
        continue
    else:
        t1 = t[-1]
    found = []
    for i, t in enumerate(t_pd0):
        if t>=t0 and t<=t1:
            found.append(i)
    if found:
        n_processed+=1
        basefn = os.path.basename(fn)
        basefn, _ = os.path.splitext(basefn)
        target_dir = args.pd0_directory
        target_file = basefn+"."+pd0_ext
        target_path = os.path.join(target_dir, target_file)
        with open(target_file, 'bw') as fpout:
            for i in found:
                with open(f_pd0[i], 'br') as fin:
                    fpout.write(fin.read())
        sys.stdout.write("%3d %s (%d)\n"%(n_processed, target_file, len(found)))




    
    
