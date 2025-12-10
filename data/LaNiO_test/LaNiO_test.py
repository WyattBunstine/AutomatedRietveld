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
from mystic.math import almostEqual
from mystic.constraints import with_mean

@with_mean(1/5.0)
def PhaseFractionConstraint(x):
    return [t/sum(x) for t in x]


class PhaseFractionCostFunction(models.AbstractFunction):

    def __init__(self, *args, ParticleSize=None, **kwargs):
        super().__init__(*args, **kwargs)

        # if True then create the subdir and copy data files there
        self.initialized = False
        self.ParticleSize = ParticleSize
        # if True then print GSAS-II output
        self.debug = False

        # define a list of detectors
        self.detectors = [dict(data_file="./ML_Poot_20241024_6_RYH_0-RYH1055F_LaNiO3_approx4gNaI_100Chrup_9hrdwell_900C_100Chrdown_ black_fluxremaining_15min.raw",
                               detector_file="./LabXRD.prm",
                               min_two_theta=5.0,
                               max_two_theta=60.0)]

        # define a list of phases
        self.phases = [dict(phase_file="./LaNiO3.cif",
                            phase_label="LaNiO3"),
                       dict(phase_file="./La2NiO4.cif",
                            phase_label="La2NiO4"),
                       dict(phase_file="./La3Ni2O7.cif",
                            phase_label="La3Ni2O7"),
                       dict(phase_file="./La4Ni3O10.cif",
                            phase_label="La4Ni3O10"),
                       dict(phase_file="./NiO.cif",
                            phase_label="NiO")
                       ]

    def function(self, p):

        # get start time of this step for stdout
        t0 = time.time()
        self.debug = False
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
        gpx = gsasii.G2Project(newgpx=f"{dir_name}/LaNiO.gpx")

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
                "no. coeffs": 15,
                "refine": True,
            }
        }
        for hist in gpx.histograms():
            hist.set_refinements(args)
        if self.ParticleSize is not None:
            for index,phase in enumerate(gpx.phases()):
                #phase.set_refinements({"Cell": True})
                phase.set_HAP_refinements({"Size":{'type':'isotropic',
                                                   'refine': False,"value":self.ParticleSize[index]}})
        # refine
        gpx.do_refinements([{}])
        gpx.save(f"{dir_name}/step_1.gpx")

        # create a GSAS-II project
        gpx = gsasii.G2Project(f"{dir_name}/step_1.gpx")
        gpx.save(f"{dir_name}/step_2.gpx")

        for index,phase in enumerate(gpx.phases()):
            #phase.set_refinements({"Cell": True})
            phase.HAPvalue("Scale",p[index])

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
        print(f"Phase percentages are {[str(x)[0:4] for x in p]}, Our R-factor is {stat} and it took {time.time() - t0}s to compute")

        return stat


class ParticleSizeCostFunction(models.AbstractFunction):

    def __init__(self, phase_fractions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_fractions = phase_fractions
        # if True then create the subdir and copy data files there
        self.initialized = False

        # if True then print GSAS-II output
        self.debug = False

        # define a list of detectors
        self.detectors = [dict(data_file="./ML_Poot_20241024_6_RYH_0-RYH1055F_LaNiO3_approx4gNaI_100Chrup_9hrdwell_900C_100Chrdown_ black_fluxremaining_15min.raw",
                               detector_file="./LabXRD.prm",
                               min_two_theta=5.0,
                               max_two_theta=60.0)]

        # define a list of phases
        self.phases = [dict(phase_file="./LaNiO3.cif",
                            phase_label="LaNiO3"),
                       dict(phase_file="./La2NiO4.cif",
                            phase_label="La2NiO4"),
                       dict(phase_file="./La3Ni2O7.cif",
                            phase_label="La3Ni2O7"),
                       dict(phase_file="./La4Ni3O10.cif",
                            phase_label="La4Ni3O10"),
                       dict(phase_file="./NiO.cif",
                            phase_label="NiO")
                       ]

    def function(self, p):

        # get start time of this step for stdout
        t0 = time.time()
        self.debug = False
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
        gpx = gsasii.G2Project(newgpx=f"{dir_name}/LaNiO.gpx")

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
                "no. coeffs": 15,
                "refine": True,
            }
        }
        for hist in gpx.histograms():
            hist.set_refinements(args)

        for index,phase in enumerate(gpx.phases()):
            #phase.set_refinements({"Cell": True})
            phase.HAPvalue("Scale",self.phase_fractions[index])

        # refine
        gpx.do_refinements([{}])
        gpx.save(f"{dir_name}/step_1.gpx")

        # create a GSAS-II project
        gpx = gsasii.G2Project(f"{dir_name}/step_1.gpx")
        gpx.save(f"{dir_name}/step_2.gpx")

        for index,phase in enumerate(gpx.phases()):
            #phase.set_refinements({"Cell": True})
            phase.set_HAP_refinements({"Size":{'type':'isotropic', 'refine': False,"value":p[index]}})

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
        print(f"Particle Sizes are {[str(x)[0:4] for x in p]}, Our R-factor is {stat} and it took {time.time() - t0}s to compute")

        return stat


# these are the target phase fractions, first is 113,214,327,4310,NiO
target = range(5)
lower_bounds = [0.0 for x in target]
upper_bounds = [1.0 for x in target]
# get number of parameters in model
ndim = len(target)
# set random seed so we can reproduce results
tools.random_seed(0)
# create a solver
solver = BuckshotSolver(dim=ndim, npts=8)
# set multi-processing pool
solver.SetMapper(Pool().map)
# since we have a search solver
# we specify what optimization algorithm to use within the search
# we tell the optimizer to not go more than 50 evaluations of our cost function
subsolver = NelderMeadSimplexSolver(ndim)
subsolver.SetEvaluationLimits(100, 100)
solver.SetNestedSolver(subsolver)
# set the range to search for all parameters
solver.SetStrictRanges(lower_bounds, upper_bounds)
solver.SetConstraints(PhaseFractionConstraint)
# find the minimum
solver.Solve(PhaseFractionCostFunction(ndim), VTR())

# print the best parameters
first_pass_phase_fraction = solver.bestSolution
print(f"\n\nThe best fp phase fraction solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n\n")


# these are the target phase fractions, first is 113,214,327,4310,NiO
#first_pass_phase_fraction = (0.57, 0.001, 0.001, 0.021, 0.09)
lower_bounds = [0.0 for x in target]
upper_bounds = [0.1 for x in target]
# get number of parameters in model
ndim = len(target)
# set random seed so we can reproduce results
tools.random_seed(0)
# create a solver
solver = BuckshotSolver(dim=ndim, npts=8)
# set multi-processing pool
solver.SetMapper(Pool().map)
# since we have a search solver
# we specify what optimization algorithm to use within the search
# we tell the optimizer to not go more than 50 evaluations of our cost function
subsolver = NelderMeadSimplexSolver(ndim)
subsolver.SetEvaluationLimits(100, 100)
solver.SetNestedSolver(subsolver)
# set the range to search for all parameters
solver.SetStrictRanges(lower_bounds, upper_bounds)
# find the minimum
solver.Solve(ParticleSizeCostFunction(first_pass_phase_fraction,ndim), VTR())
first_pass_particle_size = solver.bestSolution
print(f"\n\nThe best fp particle size solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n\n")


lower_bounds = [x*0.8 for x in first_pass_phase_fraction]
upper_bounds = [x*1.2 for x in first_pass_phase_fraction]
# get number of parameters in model
ndim = len(target)
# set random seed so we can reproduce results
tools.random_seed(0)
# create a solver
solver = BuckshotSolver(dim=ndim, npts=8)
# set multi-processing pool
solver.SetMapper(Pool().map)
# since we have a search solver
# we specify what optimization algorithm to use within the search
# we tell the optimizer to not go more than 50 evaluations of our cost function
subsolver = NelderMeadSimplexSolver(ndim)
subsolver.SetEvaluationLimits(100, 100)
solver.SetNestedSolver(subsolver)
# set the range to search for all parameters
solver.SetStrictRanges(lower_bounds, upper_bounds)
solver.SetConstraints(PhaseFractionConstraint)
# find the minimum
solver.Solve(PhaseFractionCostFunction(ndim,ParticleSize=first_pass_particle_size), VTR())

# print the best parameters
second_pass_phase_fraction = solver.bestSolution
print(f"\n\nThe best sp phase fraction solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n\n")

lower_bounds = [x*0.8 for x in first_pass_particle_size]
upper_bounds = [x*1.2 for x in first_pass_particle_size]
# get number of parameters in model
ndim = len(target)
# set random seed so we can reproduce results
tools.random_seed(0)
# create a solver
solver = BuckshotSolver(dim=ndim, npts=8)
# set multi-processing pool
solver.SetMapper(Pool().map)
# since we have a search solver
# we specify what optimization algorithm to use within the search
# we tell the optimizer to not go more than 50 evaluations of our cost function
subsolver = NelderMeadSimplexSolver(ndim)
subsolver.SetEvaluationLimits(100, 100)
solver.SetNestedSolver(subsolver)
# set the range to search for all parameters
solver.SetStrictRanges(lower_bounds, upper_bounds)
# find the minimum
solver.Solve(ParticleSizeCostFunction(second_pass_phase_fraction,ndim), VTR())
second_pass_particle_size = solver.bestSolution
print(f"\n\nThe best sp particle size solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n\n")