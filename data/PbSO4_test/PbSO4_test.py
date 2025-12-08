import sys
from GSASII import GSASIIscriptable as gsasii
from GSASII import GSASIIlattice as lattice
import io
import multiprocess
import os
import shutil
import sys
import time
from mystic import models
from spotlight import filesystem
from mystic import tools
from mystic.solvers import BuckshotSolver
from mystic.solvers import NelderMeadSimplexSolver
from mystic.termination import VTR
from pathos.pools import ProcessPool as Pool

class CostFunction(models.AbstractFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if True then create the subdir and copy data files there
        self.initialized = False

        # if True then print GSAS-II output
        self.debug = False

        # define a list of detectors
        self.detectors = [dict(data_file="./PBSO4.xra",
                               detector_file="./INST_XRY.prm",
                               min_two_theta=16.0,
                               max_two_theta=158.4),
                          dict(data_file="./PBSO4.cwn",
                               detector_file="./inst_d1a.prm",
                               min_two_theta=19.0,
                               max_two_theta=153.0)]

        # define a list of phases
        self.phases = [dict(phase_file="./PbSO4-Wyckoff.cif",
                            phase_label="PBSO4")]

    def function(self, p):

        # get start time of this step for stdout
        t0 = time.time()

        # create run dir
        dir_name = f"opt_{multiprocess.current_process().name}"
        if not self.initialized:
            filesystem.mkdir(dir_name)
            for detector in self.detectors:
                filesystem.cp([detector["data_file"], detector["detector_file"]], dest=dir_name)
            for phase in self.phases:
                filesystem.cp([phase["phase_file"]], dest=dir_name)
            self.initialized = True

        # create a text trap and redirect stdout
        # this is just to make the stdout easier to follow
        if not self.debug:
            silent_stdout = io.StringIO()
            sys.stdout = sys.stderr = silent_stdout

        # create a GSAS-II project
        gpx = gsasii.G2Project(newgpx=f"{dir_name}/lead_sulphate.gpx")

        # add histograms
        for det in self.detectors:
            gpx.add_powder_histogram(det["data_file"], det["detector_file"])

        # add phases
        for phase in self.phases:
            gpx.add_phase(phase["phase_file"], phase["phase_label"],
                          histograms=gpx.histograms())

        # turn on background refinement
        args = {
            "Background": {
                "no. coeffs": 3,
                "refine": True,
            }
        }
        for hist in gpx.histograms():
            hist.set_refinements(args)

        # refine
        gpx.do_refinements([{}])
        gpx.save(f"{dir_name}/step_1.gpx")

        # create a GSAS-II project
        gpx = gsasii.G2Project(f"{dir_name}/step_1.gpx")
        gpx.save(f"{dir_name}/step_2.gpx")

        # change lattice parameters
        for phase in gpx["Phases"].keys():

            # ignore data key
            if phase == "data":
                continue

            # handle PBSO4 phase
            elif phase == "PBSO4":
                cell = gpx["Phases"][phase]["General"]["Cell"]
                a, b, c = p
                t11, t22, t33 = cell[1] / a, cell[2] / b, cell[3] / c
                gpx["Phases"][phase]["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[t11, 0.0, 0.0],
                                [0.0, t22, 0.0],
                                [0.0, 0.0, t33]])

            # otherwise raise error because refinement plan does not support this phase
            else:
                raise NotImplementedError("Refinement plan cannot handle phase {}".format(phase))

        # turn on unit cell refinement
        args = {
            "set": {
                "Cell": True,
            }
        }

        # refine
        gpx.set_refinement(args)
        gpx.do_refinements([{}])

        # now restore stdout and stderr
        if not self.debug:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # get minimization statistic
        stat = gpx["Covariance"]["data"]["Rvals"]["Rwp"]

        # print a message to follow the results
        print(f"Our R-factor is {stat} and it took {time.time() - t0}s to compute")

        return stat

# set the ranges
target = [8.474, 5.394, 6.954]
lower_bounds = [x * 0.95 for x in target]
upper_bounds = [x * 1.05 for x in target]

# get number of parameters in model
ndim = len(target)

# set random seed so we can reproduce results
tools.random_seed(0)

# create a solver
solver = BuckshotSolver(dim=ndim, npts=8)

# set multi-processing pool
solver.SetMapper(Pool().map)

# since we have an search solver
# we specify what optimization algorithm to use within the search
# we tell the optimizer to not go more than 50 evaluations of our cost function
subsolver = NelderMeadSimplexSolver(ndim)
subsolver.SetEvaluationLimits(50, 50)
solver.SetNestedSolver(subsolver)

# set the range to search for all parameters
solver.SetStrictRanges(lower_bounds, upper_bounds)

# find the minimum
solver.Solve(CostFunction(ndim), VTR())

# print the best parameters
print(f"The best solution is {solver.bestSolution} with Rwp {solver.bestEnergy}")
print(f"The reference solutions is {target}")
ratios = [x / y for x, y in zip(target, solver.bestSolution)]
print(f"The ratios of to the reference values are {ratios}")