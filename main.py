import numpy as np
import matplotlib.pyplot as plt
import caffe_jamali as cj
import caffe_jamali_teste_sergio as cj_sergio
import plot as plot
import superficies as rf
from functions import bicubic_interpolation_opencv, extract_elev_Tiff


def main():
    n_iterations = 500

    rules = [cj.applyRules, cj_sergio.applyRules]
    rule_names = ["caffe_jamali", "caffe_jamali_sergio"]

    elevation_matrices = [rf.paraboloide()]
    elevation_names = ["paraboloide"]

    for j in range(len(rules)):
        output_dir = f"simulation_videos/{rule_names[j]}/"
        plotter = plot.DynamicPlot(rules[j], n_iterations, cmap_elev="viridis", cmap_water="Blues")
        for i in range(len(elevation_matrices)):
            n, m = elevation_matrices[i].shape
            EV = np.random.rand(n, m)*10.0 # água aleatória até 10 por célula 

            video_name = elevation_names[i]
            
            plotter.set_plot(elevation_matrices[i], EV, output_dir, video_name)
            plotter.plot()
    


if __name__ == '__main__':
    main()
