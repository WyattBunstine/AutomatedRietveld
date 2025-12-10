import sys
from GSASII import GSASIIscriptable as gsasii
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
import json
import numpy as np

#@with_mean(1/5.0)
def PhaseFractionConstraint(x):
    return [t/sum(x) for t in x]


class PhaseFractionCostFunction(models.AbstractFunction):

    def __init__(self,datafiles, phases,  *args, ParticleSize=None, **kwargs):
        super().__init__(*args, **kwargs)

        # if True then create the subdir and copy data files there
        self.initialized = False
        self.ParticleSize = ParticleSize
        # if True then print GSAS-II output
        self.debug = False

        # define a list of detectors
        self.detectors = datafiles

        # define a list of phases
        self.phases = phases


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
        gpx = gsasii.G2Project(newgpx=f"{dir_name}/initial.gpx")

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
        if False:
            print(f"Phase percentages are {[str(x)[0:4] for x in p]}, Our R-factor is {stat} and it took {time.time() - t0}s to compute")

        return stat


class ParticleSizeCostFunction(models.AbstractFunction):

    def __init__(self, datafiles, phases, *args, phase_fractions = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_fractions = phase_fractions
        # if True then create the subdir and copy data files there
        self.initialized = False

        # if True then print GSAS-II output
        self.debug = False

        # define a list of detectors
        self.detectors = datafiles

        # define a list of phases
        self.phases = phases

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
        gpx = gsasii.G2Project(newgpx=f"{dir_name}/initial.gpx")

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
        #print(f"Particle Sizes are {[str(x)[0:4] for x in p]}, Our R-factor is {stat} and it took {time.time() - t0}s to compute")

        return stat

def main():
    with open('config.json') as f:
        config = json.load(f)

    detectors = config["detectors"]
    phases = config["phases"]
    generations = config["generations"]
    evaluations = config["evaluations"]
    target = range(len(phases))
    ndim = len(target)


    optim_rounds = [{"lower_bounds":[0.01 for x in target],"upper_bounds":[1.0 for x in target],
                     "constraints": [PhaseFractionConstraint],"type":"Phase Fraction"},

                    {"lower_bounds": [0.01 for x in target], "upper_bounds": [0.2 for x in target],
                     "constraints": [], "type": "Particle Size"},

                    {"lower_bounds": "last", "upper_bounds": "last",
                     "constraints": [], "type": "Phase Fraction"},

                    {"lower_bounds": "last", "upper_bounds": "last",
                     "constraints": [], "type": "Particle Size"}
                    ]

    last_phase_fraction = None
    last_particle_size = None
    round_number = 0
    for round in optim_rounds:

        round_number +=1
        if round["lower_bounds"] == "last":
            if round["type"] == "Phase Fraction":
                lower_bounds = [x*0.8 for x in last_phase_fraction]
            else:
                lower_bounds = [x*0.8 for x in last_particle_size]
        else:
            lower_bounds = round["lower_bounds"]
        if round["upper_bounds"] == "last":
            if round["type"] == "Phase Fraction":
                upper_bounds = [x * 1.2 for x in last_phase_fraction]
            else:
                upper_bounds = [x * 1.2 for x in last_particle_size]
        else:
            upper_bounds = round["upper_bounds"]


        tools.random_seed(0)
        solver = BuckshotSolver(dim=ndim, npts=8)
        solver.SetMapper(Pool().map)
        subsolver = NelderMeadSimplexSolver(ndim)
        subsolver.SetEvaluationLimits(generations, evaluations)
        solver.SetNestedSolver(subsolver)
        # set the range to search for all parameters
        solver.SetStrictRanges(lower_bounds, upper_bounds)
        for constraint in round["constraints"]:
            solver.SetConstraints(constraint)
        # find the minimum
        if round["type"] == "Phase Fraction":
            solver.Solve(PhaseFractionCostFunction(detectors,phases,ndim,ParticleSize=last_particle_size), VTR())
            last_phase_fraction = solver.bestSolution
        if round["type"] == "Particle Size":
            solver.Solve(ParticleSizeCostFunction(detectors,phases,ndim,
                                                  phase_fractions = last_phase_fraction), VTR())
            last_particle_size = solver.bestSolution

        print(
            f"\n" + "Round: " + str(round_number) + ". The best " + round["type"] +
            f" solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n\n")


    return

    # these are the target phase fractions, first is 113,214,327,4310,NiO
    target = range(len(phases))
    lower_bounds = [0.0 for x in target]
    upper_bounds = [1.0 for x in target]
    lower_bounds[0] = 0.1
    ndim = len(target)
    tools.random_seed(0)
    solver = BuckshotSolver(dim=ndim, npts=8)
    solver.SetMapper(Pool().map)
    subsolver = NelderMeadSimplexSolver(ndim)
    subsolver.SetEvaluationLimits(generations, evaluations)
    solver.SetNestedSolver(subsolver)
    # set the range to search for all parameters
    solver.SetStrictRanges(lower_bounds, upper_bounds)
    solver.SetConstraints(PhaseFractionConstraint)
    # find the minimum
    solver.Solve(PhaseFractionCostFunction(detectors,phases,ndim), VTR())

    # print the best parameters
    first_pass_phase_fraction = solver.bestSolution
    #first_pass_phase_fraction = np.array(first_pass_phase_fraction)/sum(first_pass_phase_fraction)
    print(f"\n\nThe best fp phase fraction solution is {first_pass_phase_fraction} with Rwp {solver.bestEnergy}\n\n")


    lower_bounds = [0.0 for x in target]
    upper_bounds = [0.2 for x in target]
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
    subsolver.SetEvaluationLimits(generations, evaluations)
    solver.SetNestedSolver(subsolver)
    # set the range to search for all parameters
    solver.SetStrictRanges(lower_bounds, upper_bounds)
    # find the minimum
    solver.Solve(ParticleSizeCostFunction(detectors,phases,ndim,phase_fractions=first_pass_phase_fraction), VTR())
    first_pass_particle_size = solver.bestSolution
    print(f"\n\nThe best fp particle size solution is {first_pass_particle_size} with Rwp {solver.bestEnergy}\n\n")


    tmp = [float(x.item())-0.1 for x in first_pass_phase_fraction]
    lower_bounds = []
    for x in tmp:
        if x < 0:
            lower_bounds.append(0.05)
        else:
            lower_bounds.append(x)
    tmp = [float(x.item())+0.1 for x in first_pass_phase_fraction]
    upper_bounds = []
    for x in tmp:
        if x > 1.0:
            upper_bounds.append(1.0)
        else:
            upper_bounds.append(x)
    lower_bounds = [x*0.9 for x in first_pass_phase_fraction]
    upper_bounds = [x*1.1 for x in first_pass_phase_fraction]
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
    subsolver.SetEvaluationLimits(generations, evaluations)
    solver.SetNestedSolver(subsolver)
    # set the range to search for all parameters
    solver.SetStrictRanges(lower_bounds, upper_bounds)
    solver.SetConstraints(PhaseFractionConstraint)
    # find the minimum
    solver.Solve(PhaseFractionCostFunction(detectors,phases,ndim,ParticleSize=first_pass_particle_size), VTR())
    # print the best parameters
    second_pass_phase_fraction = solver.bestSolution
    print(f"\n\nThe best sp phase fraction solution is {second_pass_phase_fraction} with Rwp {solver.bestEnergy}\n\n")

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
    subsolver.SetEvaluationLimits(generations, evaluations)
    solver.SetNestedSolver(subsolver)
    # set the range to search for all parameters
    solver.SetStrictRanges(lower_bounds, upper_bounds)
    # find the minimum
    solver.Solve(ParticleSizeCostFunction(detectors,phases,ndim, phase_fractions = second_pass_phase_fraction), VTR())
    second_pass_particle_size = solver.bestSolution
    print(f"\n\nThe best sp particle size solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n\n")


if __name__ == "__main__":
    main()

