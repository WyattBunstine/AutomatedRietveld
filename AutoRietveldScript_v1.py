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
from mystic.solvers import BuckshotSolver, DifferentialEvolutionSolver2
from mystic.solvers import NelderMeadSimplexSolver
from mystic.termination import *
from pathos.pools import ProcessPool as Pool
from mystic.math import almostEqual
from mystic.constraints import with_mean
from mystic.monitors import VerboseMonitor, Monitor
import json
import numpy as np

Print = True
spacer = "########################################################################################################################"

#@with_mean(1/5.0)
def PhaseFractionConstraint(x):
    return [t/sum(x) for t in x]

def generate_project(datafiles, phases, lattice_parameters = None, phase_fractions=None,particle_sizes=None,
                     ret_val = None, file_name = "gsas_proj"):
    gpx = gsasii.G2Project(newgpx=f"{file_name}.gpx")
    for detector in datafiles:
        gpx.add_powder_histogram(detector["data_file"], detector["detector_file"])
    for phase in phases:
        gpx.add_phase(phase["phase_file"], phase["phase_label"], histograms=gpx.histograms())

    # turn on background refinement
    args = {
        "Background": {
            "no. coeffs": 15,
            "refine": True,
        }
    }
    for hist in gpx.histograms():
        hist.set_refinements(args)

    # refine
    gpx.do_refinements([{}])


    for index, phase in enumerate(gpx.phases()):
        if lattice_parameters is not None:
            cell = phase["General"]["Cell"]
            phase["General"]["Cell"][1:] = lattice.TransformCell(
                cell[1:7], [[lattice_parameters[index][0], 0.0, 0.0],
                            [0.0, lattice_parameters[index][1], 0.0],
                            [0.0, 0.0, lattice_parameters[index][2]]])
        if particle_sizes is not None:
            phase.set_HAP_refinements({"Size": {'type': 'isotropic', 'refine': False, "value": particle_sizes[index]}})
        if phase_fractions is not None:
            phase.HAPvalue("Scale", phase_fractions[index])
        phase.set_refinements({"Cell": True})
    gpx.do_refinements([{}])
    gpx.save(f"{file_name}.gpx")
    if ret_val is None:
        return gpx


class LatticeParameterCostFunction(models.AbstractFunction):

    def __init__(self, datafiles, phases,  *args, optim_phase = None, non_opt_lat_param = None, **kwargs):
        super().__init__(*args, **kwargs)

        # if True then create the subdir and copy data files there
        self.initialized = False
        # if True then print GSAS-II output
        self.debug = False

        self.non_opt_lat_param = non_opt_lat_param

        self.optim_phase = optim_phase
        if optim_phase is None:
            self.optim_phase = phases[0]["phase_label"]

        # define a list of detectors
        self.detectors = datafiles

        # define a list of phases
        self.phases = phases

        self.refine_particle_size = False



    def function(self, p, return_lattice_params=False):
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

        # refine
        gpx.do_refinements([{}])
        #gpx.save(f"{dir_name}/step_1.gpx")

        # create a GSAS-II project
        #gpx = gsasii.G2Project(f"{dir_name}/step_1.gpx")
        #gpx.save(f"{dir_name}/step_2.gpx")

        for index,phase in enumerate(gpx.phases()):
            if self.optim_phase is None or phase.name == self.optim_phase:
                prev_lattice_params = []
                [(prev_lattice_params.append(phase.get_cell()[x]) if "len" in x else 1) for x in phase.get_cell().keys()]
                cell = phase["General"]["Cell"]
                phase["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[p[0], 0.0, 0.0],
                                [0.0, p[1], 0.0],
                                [0.0, 0.0, p[2]]])
                lattice_params = []
                [(lattice_params.append(phase.get_cell()[x]) if "len" in x else 1) for x in phase.get_cell().keys()]
                phase.set_HAP_refinements({"Size": {'type': 'isotropic', 'refine': self.refine_particle_size, "value": 5.0}})
                phase.set_refinements({"Cell": True})
            else:
                tranform = self.non_opt_lat_param[phase.name]
                cell = phase["General"]["Cell"]
                phase["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[tranform[0], 0.0, 0.0],
                                [0.0, tranform[1], 0.0],
                                [0.0, 0.0, tranform[2]]])
                phase.set_HAP_refinements({"Size": {'type': 'isotropic', 'refine': self.refine_particle_size, "value": 5.0}})
                phase.set_refinements({"Cell": False})
        # refine
        gpx.do_refinements([{}])

        # now restore stdout and stderr
        if not self.debug:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # get minimization statistic
        stat = gpx["Covariance"]["data"]["Rvals"]["Rwp"]
        if Print:

            print(
                f"Lattice Parameters for " + self.optim_phase +f" are {[str(x)[0:5] for x in lattice_params]} , Our R-factor is {stat}")
        if return_lattice_params:
            for index, phase in enumerate(gpx.phases()):
                if self.optim_phase is None or phase.name == self.optim_phase:
                    lattice_params = []
                    [(lattice_params.append(phase.get_cell()[x]) if "len" in x else 1) for x in phase.get_cell().keys()]
                    gpx.save(f"{dir_name}/{self.optim_phase}_lat_params.gpx")
                    return lattice_params, [prev_lattice_params[i]/lattice_params[i] for i in range(len(lattice_params))]
        return stat

class PhaseFractionCostFunction(models.AbstractFunction):

    def __init__(self,datafiles, phases,  *args, ParticleSize=None,lattice_parameters=None, **kwargs):
        super().__init__(*args, **kwargs)

        # if True then create the subdir and copy data files there
        self.initialized = False
        self.ParticleSize = ParticleSize
        self.lattice_parameters = lattice_parameters
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

        if self.lattice_parameters is not None:
            for index, phase in enumerate(gpx.phases()):
                parameters = self.lattice_parameters[index]
                cell = phase["General"]["Cell"]

                phase["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[parameters[0], 0.0, 0.0],
                                [0.0, parameters[1], 0.0],
                                [0.0, 0.0, parameters[2]]])

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
                "Cell": False,
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
        if Print:
            print(f"Phase percentages are {[str(x)[0:4] for x in p]}, Our R-factor is {stat} and it took {time.time() - t0}s to compute")

        return stat

class ParticleSizeCostFunction(models.AbstractFunction):

    def __init__(self, datafiles, phases, *args, phase_fractions = None,lattice_parameters=None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_fractions = phase_fractions
        self.lattice_parameters = lattice_parameters
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

        if self.phase_fractions is not None:
            for index,phase in enumerate(gpx.phases()):
                #phase.set_refinements({"Cell": True})
                phase.HAPvalue("Scale",self.phase_fractions[index])

        if not self.lattice_parameters is None:
            for index, phase in enumerate(gpx.phases()):
                parameters = self.lattice_parameters[index]
                cell = phase["General"]["Cell"]
                phase["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[parameters[0], 0.0, 0.0],
                                [0.0, parameters[1], 0.0],
                                [0.0, 0.0, parameters[2]]])

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
                "Cell": False,
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
        if Print:
            print(f"Particle Sizes are {[str(x)[0:4] for x in p]}, Our R-factor is {stat} and it took {time.time() - t0}s to compute")

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

    #Start by optimizing lattice parameters of each phase contribution

    lat_dim = 3
    lower_bounds = [0.9 for x in range(lat_dim)]
    upper_bounds = [1.1 for x in range(lat_dim)]
    fit_phases = []
    fit_latt_params = []

    print(spacer +"\nStarting lattice parameter optimization\n"+spacer)
    if True:
        for outer_index,outer_phase in enumerate(phases):
            non_opt_lat_param = {}
            fit_phase_dicts = []
            for index,optimized_phase in enumerate(fit_phases):
                non_opt_lat_param[optimized_phase] = fit_latt_params[index]
            for phase in phases:
                if phase["phase_label"] in fit_phases:
                    fit_phase_dicts.append(phase)

            best_phase_solution = None
            best_phase = None
            lattice_parameters = []
            for index, phase in enumerate(phases):
                t0 = time.time()
                if phase["phase_label"] in fit_phases:
                    lattice_parameters.append(None)
                    continue
                # configure monitor
                tools.random_seed(0)
                # create a solver
                solver = BuckshotSolver(dim=lat_dim, npts=12)
                # set multi-processing pool
                solver.SetMapper(Pool().map)
                # since we have a search solver
                # we specify what optimization algorithm to use within the search
                # we tell the optimizer to not go more than 50 evaluations of our cost function
                subsolver = DifferentialEvolutionSolver2(lat_dim,10*lat_dim)
                subsolver.SetRandomInitialPoints(min=[0.9] * lat_dim, max=[1.1] * lat_dim)
                subsolver.SetTermination(ChangeOverGeneration(1,10))
                subsolver.SetEvaluationLimits(1000,1000)
                #subsolver.SetEvaluationLimits(generations, evaluations)
                solver.SetNestedSolver(subsolver)

                # set the range to search for all parameters
                solver.SetStrictRanges(lower_bounds, upper_bounds)
                # find the minimum
                solver.Solve(LatticeParameterCostFunction(detectors, fit_phase_dicts + [phase], lat_dim,
                                                          non_opt_lat_param=non_opt_lat_param, optim_phase=phase["phase_label"]),
                             ChangeOverGeneration(1,10))
                print(
                    f"" + str(phase["phase_label"]) + f" lattice parameters : {solver.bestSolution} with Rwp {solver.bestEnergy}, optimization time: {time.time()-t0} seconds.\n")
                lattice_parameters.append(solver.bestSolution)
                if best_phase_solution is None or solver.bestEnergy < best_phase_solution:
                    best_phase_solution = solver.bestEnergy
                    best_phase = index

            gsasproj = LatticeParameterCostFunction(detectors, fit_phase_dicts + [phases[best_phase]], lat_dim,
                                         non_opt_lat_param=non_opt_lat_param, optim_phase=phases[best_phase]["phase_label"])
            refined_lattice_parameters, refined_scaling = gsasproj.function(lattice_parameters[best_phase],return_lattice_params=True)

            print("Round " + str(outer_index+1) + " best phase is " + phases[best_phase]["phase_label"])
            fit_phases.append(phases[best_phase]["phase_label"])
            fit_latt_params.append(refined_scaling)
            print("fit phases are " + str(fit_phases))
        print(spacer + "\nFinal lattice parameters are " + str(fit_latt_params) + " for phases " + str(fit_phases)+"\n"+spacer)

        print(spacer + "\nStarting phase fraction and particle size optimization\n"+spacer)
    else:
        fit_latt_params = [[0.99888128, 1.00831389, 1.00152381], [0.99996671, 0.90257233, 0.94937643]]
    generate_project(detectors,phases,fit_latt_params)
    1/0
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

        stepmon = VerboseMonitor(50)
        solver = DifferentialEvolutionSolver2(ndim, 10 * ndim)
        solver.SetRandomInitialPoints(min=[0.0] * ndim, max=[1.0] * ndim)
        solver.SetGenerationMonitor(stepmon)
        solver.enable_signal_handler()
        solver.SetStrictRanges(lower_bounds, upper_bounds)
        # find the minimum
        for constraint in round["constraints"]:
            solver.SetConstraints(constraint)
        # find the minimum
        if round["type"] == "Phase Fraction":
            solver.Solve(PhaseFractionCostFunction(detectors,phases,ndim,ParticleSize=last_particle_size,
                                                   lattice_parameters=fit_latt_params), VTRChangeOverGeneration(gtol=0.5))
            last_phase_fraction = solver.bestSolution
        if round["type"] == "Particle Size":
            solver.Solve(ParticleSizeCostFunction(detectors,phases,ndim,
                                                  phase_fractions = last_phase_fraction,lattice_parameters=fit_latt_params), SolutionImprovement(0.001))
            last_particle_size = solver.bestSolution

        print(
            f"" + "Round: " + str(round_number) + ". The best " + round["type"] +
            f" solution is {solver.bestSolution} with Rwp {solver.bestEnergy}\n")
    print(spacer + "\nFinal phase fractions are " + str(last_phase_fraction) +
          " and particle sizes are " + str(last_particle_size) + "\n"+spacer)


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


    #####################

    stepmon = VerboseMonitor(50)
    solver = DifferentialEvolutionSolver2(lat_dim, 10 * lat_dim)
    solver.SetRandomInitialPoints(min=[0.9] * lat_dim, max=[1.1] * lat_dim)
    solver.SetGenerationMonitor(stepmon)
    solver.enable_signal_handler()
    solver.SetStrictRanges(lower_bounds, upper_bounds)
    # find the minimum
    solver.Solve(LatticeParameterCostFunction(detectors, fit_phases + [phase], lat_dim,
                                              non_opt_lat_param=non_opt_lat_param, optim_phase=phase["phase_label"]),
                 VTRChangeOverGeneration(gtol=1.0))


if __name__ == "__main__":
    main()

