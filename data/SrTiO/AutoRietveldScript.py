import sys
from GSASII import GSASIIscriptable as gsasii
from GSASII import GSASIIlattice as lattice
from GSASII import GSASIIpwdplot
import io
import multiprocess
import multiprocessing
import os
import shutil
import sys
import time
from mystic import models
from spotlight import filesystem
import matplotlib.pyplot as plt

import json
import numpy as np

Print = False
spacer = "########################################################################################################################"

#@with_mean(1/5.0)
def PhaseFractionConstraint(x):
    return [t/sum(x) for t in x]

def generate_project(datafiles, phases, lattice_parameters = None, fit_phases=None, file_name = "gsas_proj", debug=False):
    if not debug:
        silent_stdout = io.StringIO()
        sys.stdout = sys.stderr = silent_stdout

    gpx = gsasii.G2Project(newgpx=f"{file_name}.gpx")
    for detector in datafiles:
        gpx.add_powder_histogram(detector["data_file"], detector["detector_file"])
    for phase in phases:
        gpx.add_phase(phase["phase_file"], phase["phase_label"], histograms=gpx.histograms())
    fit_phase_names = []
    for phase in fit_phases:
        gpx.add_phase(phasename=phase.name, histograms=gpx.histograms())
        fit_phase_names.append(phase.name)
    for index, phase in enumerate(gpx.phases()):
        if index >= len(phases):
            for key in fit_phases[index - len(phases)].keys():
                phase[key] = fit_phases[index - len(phases)][key]
            phase.set_HAP_refinements({"Size": {'type': 'isotropic', "Scale": False,
                                                'refine': False, "value": 1.0}, "Pref.Ori.": False})
            phase.set_refinements({"Cell": False})


    # turn on background refinement
    args = {
        "Background": {
            "no. coeffs": 15,
            "refine": True,
        }
    }
    for hist in gpx.histograms():
        hist.set_refinements(args)
    gpx.do_refinements([{}])

    for index, phase in enumerate(gpx.phases()):
        if index < len(phases):
            if lattice_parameters is not None:
                cell = phase["General"]["Cell"]
                phase["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[lattice_parameters[index][0], 0.0, 0.0],
                                [0.0, lattice_parameters[index][1], 0.0],
                                [0.0, 0.0, lattice_parameters[index][2]]])
            phase.set_HAP_refinements({"Pref.Ori.": False})
            phase.HAPvalue("PO", 16)
            if index != 0:
                phase.set_HAP_refinements({"Scale": True})
            phase.set_refinements({"Cell": False})
    print(spacer + " \nPhase Fraction Optimization\n")
    gpx.do_refinements([{}])

    for index, phase in enumerate(gpx.phases()):
        phase.set_HAP_refinements({"Scale": False})
        if index < len(phases):
            phase.set_HAP_refinements({"Scale": False,  "Size":{'type': 'isotropic', 'refine': True, "value": 1.0},
                                       "Pref.Ori.": True})
    print(spacer + " \nParticle Size Optimization\n")
    gpx.do_refinements([{}])
    print(spacer + " \nLattice Vector Optimization\n")
    for index, phase in enumerate(gpx.phases()):
        if index < len(phases):
            phase.set_HAP_refinements({"Scale": False, "Size": {'type': 'isotropic', 'refine': False,
                                                                "value": phase.getHAPvalues(0)["Size"][1][0]},
                                       "Pref.Ori.": False})
            phase.set_refinements({"Cell": True})
    gpx.do_refinements([{}])
    print(spacer + " \nPhase Fraction Optimization\n")
    for index, phase in enumerate(gpx.phases()):
        phase.set_HAP_refinements({"Pref.Ori.": False})
        if index != len(gpx.phases()) - 1:
            phase.set_HAP_refinements({"Scale": True})
        phase.set_refinements({"Cell": False})
    gpx.do_refinements([{}])
    gpx.save(f"{file_name}.gpx")

    if not debug:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return gpx


class LatticeParameterCostFunction(models.AbstractFunction):

    def __init__(self, datafiles, phase,  *args, non_opt_phases = None, **kwargs):
        super().__init__(*args, **kwargs)

        # if True then create the subdir and copy data files there
        self.initialized = False
        # if True then print GSAS-II output
        self.debug = False

        self.non_opt_phases = non_opt_phases
        if self.non_opt_phases is None:
            self.non_opt_phases = []

        self.optim_phase = phase

        # define a list of detectors
        self.detectors = datafiles

        self.refine_particle_size = False



    def function(self, p, return_lattice_params=False):
        # get start time of this step for stdout
        t0 = time.time()
        # create run dir
        dir_name = f"workers/opt_{multiprocess.current_process().name}"
        if not self.initialized:
            filesystem.mkdir(dir_name)
            for detector in self.detectors:
                filesystem.cp([detector["data_file"], detector["detector_file"]], dest=dir_name)
            filesystem.cp([self.optim_phase["phase_file"]], dest=dir_name)
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

        # add optimization phase phases
        gpx.add_phase(self.optim_phase["phase_file"], self.optim_phase["phase_label"],histograms=gpx.histograms())

        for phase in self.non_opt_phases:
            gpx.add_phase(phasename=phase.name,histograms=gpx.histograms())
        for index, phase in enumerate(gpx.phases()):
            if index != 0:
                for key in self.non_opt_phases[index-1].keys():
                    phase[key] = self.non_opt_phases[index-1][key]
                phase.set_HAP_refinements({"Size": {'type': 'isotropic',"Scale": False,
                                                    'refine': False, "value": 1.0},"Pref.Ori.": False})
                phase.set_refinements({"Cell": False})


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
        for index,phase in enumerate(gpx.phases()):
            if phase.name == self.optim_phase["phase_label"]:
                prev_lattice_params = []
                [(prev_lattice_params.append(phase.get_cell()[x]) if "len" in x else 1) for x in phase.get_cell().keys()]
                cell = phase["General"]["Cell"]
                phase["General"]["Cell"][1:] = lattice.TransformCell(
                    cell[1:7], [[p[0], 0.0, 0.0],
                                [0.0, p[1], 0.0],
                                [0.0, 0.0, p[2]]])
                lattice_params = []
                [(lattice_params.append(phase.get_cell()[x]) if "len" in x else 1) for x in phase.get_cell().keys()]
                phase.set_HAP_refinements({"Size": {'type': 'isotropic', 'refine': self.refine_particle_size, "value": 0.1}})
                phase.set_refinements({"Cell": True})
                phase.set_HAP_refinements({"Scale": False})
        gpx.do_refinements([{}])
        for index, phase in enumerate(gpx.phases()):
            if len(gpx.phases()) > 1 and phase.name == self.optim_phase["phase_label"]:
                phase.set_refinements({"Cell": False})
                phase.set_HAP_refinements({"Scale": True})
        gpx.do_refinements([{}])
        # now restore stdout and stderr
        if not self.debug:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # get minimization statistic
        stat = gpx["Covariance"]["data"]["Rvals"]["Rwp"]
        if Print:

            print(
                f"Lattice Parameters for " + self.optim_phase["phase_label"] +f" are {[str(x)[0:5] for x in lattice_params]} , Our R-factor is {stat}")
        if return_lattice_params:
            for index, phase in enumerate(gpx.phases()):
                if self.optim_phase is None or phase.name == self.optim_phase["phase_label"]:
                    lattice_params = []
                    [(lattice_params.append(phase.get_cell()[x]) if "len" in x else 1) for x in phase.get_cell().keys()]
                    gpx.save(f"{dir_name}/{self.optim_phase["phase_label"]}_lat_params.gpx")
                    return lattice_params, [lattice_params[i]/prev_lattice_params[i] for i in range(len(lattice_params))]
        for index, phase in enumerate(gpx.phases()):
            if len(gpx.phases()) > 1 and phase.name == self.optim_phase["phase_label"]:
                if phase.HAPvalue("Scale") < 0.01:
                    return 100

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
    #Start by optimizing lattice parameters of each phase contribution

    lat_dim = 3

    fit_phases = []
    fit_latt_params = []
    lattice_param_samping = np.arange(0.98, 1.02, 0.0025)

    print(spacer +"\nStarting lattice parameter optimization\n"+spacer)
    if True:
        best_total_solution = None
        fit_phases_objects = []
        for outer_index,outer_phase in enumerate(phases):
            best_phase_solution = None
            best_phase = None
            lattice_parameters = []
            for index, phase in enumerate(phases):
                t0 = time.time()
                if phase["phase_label"] in fit_phases:
                    lattice_parameters.append(None)
                    continue

                optimfunc = LatticeParameterCostFunction(detectors, phase, lat_dim,
                                                          non_opt_phases=fit_phases_objects)
                def multiloop(x, optimfunc, return_vals):
                    #print("started process " + multiprocess.current_process().name)
                    best_val = optimfunc.function([x, 1.1, 1])
                    best_lattice_params = [x, 1.1, 1]
                    for y in [1]:#np.arange(0.9, 1.1, 0.05):
                        for z in lattice_param_samping:
                            val = optimfunc.function([x,y,z])
                            if val < best_val:
                                best_val = val
                                best_lattice_params = [x,y,z]
                    return_vals[multiprocess.current_process().name] = [best_val, best_lattice_params]
                    #print("ended process " + multiprocess.current_process().name)

                manager = multiprocessing.Manager()
                processes = []
                return_vals = manager.dict()
                for x in lattice_param_samping:
                    p = multiprocess.Process(target=multiloop, args=(x,optimfunc, return_vals))
                    p.start()
                    processes.append(p)
                best_val = optimfunc.function([1.1, 1.1, 1.1])
                best_lattice_params = [1.1, 1.1, 1.1]
                for p in processes:
                    p.join()
                for val in return_vals.keys():
                    if return_vals[val][0] < best_val:
                        best_val = return_vals[val][0]
                        best_lattice_params = return_vals[val][1]

                print(
                    f"" + str(phase["phase_label"]) + f" lattice parameters : {best_lattice_params} with Rwp {best_val}, optimization time: {time.time()-t0} seconds.\n")
                lattice_parameters.append(best_lattice_params)
                if best_phase_solution is None or best_val < best_phase_solution:
                    best_phase_solution = best_val
                    best_phase = index


            gsasproj = LatticeParameterCostFunction(detectors, phases[best_phase], lat_dim,
                                         non_opt_phases=fit_phases_objects)
            refined_lattice_parameters, refined_scaling = gsasproj.function(lattice_parameters[best_phase],return_lattice_params=True)

            print("Round " + str(outer_index+1) + " best phase is " + phases[best_phase]["phase_label"])
            print("Lattice Parameters are : "+str(refined_lattice_parameters) + " with scaling " + str(refined_scaling))
            fit_phases.append(phases[best_phase]["phase_label"])
            fit_latt_params.append(refined_scaling)
            print("fit phases are " + str(fit_phases))
            proj = generate_project(detectors, [phases[best_phase]], lattice_parameters=[refined_scaling],
                                    fit_phases=fit_phases_objects)

            if best_total_solution is not None and proj["Covariance"]["data"]["Rvals"]["Rwp"] > best_total_solution:
                print("Early stopping because no additional improvement in Rwp\nPhases added " + str(fit_phases))
                break
            fit_phases_objects = proj.phases()
            best_total_solution = proj["Covariance"]["data"]["Rvals"]["Rwp"]

        print(spacer + "\nFinal lattice parameters are " + str(fit_latt_params) + " for phases " + str(fit_phases)+"\n"+spacer)

        print(spacer + "\nStarting phase fraction and particle size optimization\n"+spacer)
        proj = generate_project(detectors, [], fit_phases=fit_phases_objects, file_name=detectors[0]["data_file"][:-4])
    else:
        proj = gsasii.G2Project(detectors[0]["data_file"][:-4] + ".gpx")

    """
    Opens a GSAS-II project file and plots the powder diffraction data 
    (observed, calculated, and difference).
    """
    # Find the specific histogram
    hist = None
    for h in proj.histograms():
        hist = h

    # Extract the data
    # xy_data is a dictionary containing 'x' (2-theta or Q), 'y' (intensity),
    # 'calc' (calculated intensity), 'bkg' (background), etc.

    x_data = hist.getdata('X')
    observed_y = hist.getdata('Yobs')
    calculated_y = hist.getdata('Ycalc')
    background_y = hist.getdata('Background')
    # The difference curve is typically the observed minus calculated
    difference_y = observed_y - calculated_y
    reflections = hist.reflections()
    weights = hist.ComputeMassFracs()
    scaling_factor = np.max(observed_y)
    # Plotting using Matplotlib
    plt.figure(figsize=(5, 5))
    plt.plot(x_data, observed_y/scaling_factor, 'ko', markersize=2, label='Observed')
    plt.plot(x_data, calculated_y/scaling_factor, 'r-', linewidth=1, label='Calculated ' + str(proj["Covariance"]["data"]["Rvals"]["Rwp"])[0:5])
    plt.plot(x_data, background_y/scaling_factor, 'b--', linewidth=1, label='Background')
    # Plot the difference curve offset below the main pattern
    offset = np.min(observed_y) * 0.8
    plt.plot(x_data, (difference_y + offset)/scaling_factor, 'g-', linewidth=1, label='Difference')

    y = np.min((difference_y + offset)/scaling_factor)
    for key in reflections.keys():
        y = y-0.1
        ticks = [reflections[key]['RefList'][x][5] for x in range(len(reflections[key]['RefList']))]
        plt.scatter(ticks,y*np.ones(len(ticks)), s=100, marker="|", label=key )#+ " " + str(weights[key][0]*100)[0:5] + "%")
    plt.plot()

    #plt.axes().get_yaxis().set_visible(False)
    plt.xlabel(r'2$\theta$ (deg)')
    plt.ylabel(f'Intensity (arb)')
    plt.title(f'{detectors[0]["data_file"]} Rwp: {proj["Covariance"]["data"]["Rvals"]["Rwp"]}')
    plt.legend()
    #plt.grid(True)
    plt.savefig(detectors[0]["data_file"][:-4]+".png",dpi=300)
    plt.show()

    return


if __name__ == "__main__":
    main()

