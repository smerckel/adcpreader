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

parser.add_argument("-o","--operation", choices="copy move symlink".split(), help = "Operation to be carried out on the original file(s).", default = "copy")


args = parser.parse_args()

pd0_ext = args.pd0_ext
dbd_ext = args.dbd_ext
operation = args.operation

for s in [args.dbd_directory, args.pd0_directory]:
    if not os.path.exists(s):
        raise IOError('"{}" does not exist.'.format(s))
    if not os.path.isdir(s):
        raise IOError('"{}" is not an directory.'.format(s))

fns = dbdreader.DBDList(glob.glob(os.path.join(args.dbd_directory,"{}*.{}".format(args.glider, dbd_ext))))
if not fns:
    raise ValueError("No {} files were found in {}.".format(iargs.dbd_ext, args.dbd_directory))
fns.sort()


pd0s = glob.glob(os.path.join(args.pd0_directory,"*.{}".format(pd0_ext)))
if not pd0s:
    raise ValueError("No {} files were found in {}.".format(args.pd0_ext, args.pd0_directory))
pd0s.sort()

t_pd0 = []
f_pd0 = []

pd0 = rdi.rdi_reader.PD0()
for fn in pd0s:
    for ens in pd0.ensemble_generator((fn,)):
        rtc = list(ens['variable_leader']['RTC'])
        rtc[0]+=2000
        rtc[6]*=1000
        tm = datetime(*rtc, timezone.utc).timestamp()
        break
    t_pd0.append(tm)
    f_pd0.append(fn)

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
    found = False
    for i, t in enumerate(t_pd0):
        if t>=t0 and t<=t1:
            found = True
            break
    if found:
        basefn = os.path.basename(fn)
        basefn, _ = os.path.splitext(basefn)
        basename_pd0 = os.path.basename(f_pd0[i])
        basename_pd0, ext_pd0 = os.path.splitext(basename_pd0)
        fn_new = f_pd0[i].replace(basename_pd0, basefn)
        if f_pd0[i]  == fn_new:
            # source and destination are the same. Silently ignore.
            continue
        n_processed += 1
        if operation == "copy":
            sys.stderr.write("Copying {} -> {} ...\n".format(os.path.basename(f_pd0[i]),
                                                             os.path.basename(fn_new)))
            shutil.copyfile(f_pd0[i], fn_new)
        elif operation == "move":
            sys.stderr.write("Moving {} -> {} ...\n".format(os.path.basename(f_pd0[i]),
                                                            os.path.basename(fn_new)))
            shutil.move(f_pd0[i], fn_new)
        elif operation == "symlink":
            sys.stderr.write("Symlinking {} -> {} ...\n".format(os.path.basename(f_pd0[i]),
                                                                os.path.basename(fn_new)))
            shutil.os.symlink(f_pd0[i], fn_new)
        else:
            raise NotImplementedError
ops = dict(copy="copied", move="moved", symlink="symlinked")
sys.stderr.write("Successfully {} {} file(s).\n".format(ops[operation], n_processed))




    
    
