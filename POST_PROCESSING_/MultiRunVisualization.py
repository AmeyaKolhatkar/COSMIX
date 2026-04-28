# Multi Run Visualization layer

from getdist import MCSamples, plots
import matplotlib.pyplot as plt
import logging
import numpy as np

class MultiRunVisualization:
    
    def __init__(self, results_list, labels):
        self.results_list = results_list
        self.labels = labels

        #self._check_param_ordering()
        self.samples_list = self._build_samples_list()
   
            
    def _build_samples_list(self):
        samples_list = []
        _log = logging.getLogger()
        _prev_level = _log.level
        _log.setLevel(logging.ERROR)
        for r in self.results_list:
            s = MCSamples(samples=r.chain,
                          names=r.param_names,
                          labels=r.latex_names,
                          weights=getattr(r, "weights", None))
            s.updateSettings({'fine_bins': 150, 'fine_bins_2D': 50})
            samples_list.append(s)
        _log.setLevel(_prev_level)
        return samples_list

    def corner_overlay(self, colors=None, filled=None, contour_ls=None, contour_lws=None, params=None, param_limits={},
                       plot_contours=2, alpha_filled_add=0.5, palefactor=0.25, legend_fontsize=15, axes_fontsize=15,
                       lw=1.8, lw_contour=2.5, label_fontsize=15, figure_legend_frame=False,
                       width_inch=8, scaling=False, rc_sizes=True):
        #plt.figure()
        if colors is not None and len(colors) != len(self.samples_list):
            raise ValueError("Number of color arguments must match number of sample sets.")
        if filled is not None and len(filled) != len(self.samples_list):
            raise ValueError("Number of filled arguments must match the number of sample sets.")
        if contour_ls is not None and len(contour_ls) != len(self.samples_list):
            raise ValueError("Number of line style arguments must match the number of sample sets.")
        if contour_lws is not None and len(contour_lws) != len(self.samples_list):
            raise ValueError("Number of line width arguments must match the number of sample sets.")
        
        g = plots.get_subplot_plotter(width_inch=width_inch, scaling=scaling, rc_sizes=rc_sizes)
        g.settings.num_plot_contours = plot_contours
        g.settings.alpha_filled_add = alpha_filled_add
        g.settings.solid_contour_palefactor = palefactor
        g.settings.linewidth = lw
        g.settings.linewidth_contour = lw_contour
        g.settings.legend_fontsize = legend_fontsize
        g.settings.axes_fontsize = axes_fontsize
        g.settings.lab_fontsize = label_fontsize
        g.settings.figure_legend_frame = figure_legend_frame
        _log = logging.getLogger()
        _prev_level = _log.level
        _log.setLevel(logging.ERROR)
        g.triangle_plot(
            self.samples_list, legend_loc="upper right",
            legend_labels=self.labels,
            contour_colors=colors,
            filled=filled,
            contour_ls=contour_ls,
            contour_lws=contour_lws,
            tight_layout=True,
            params=params,
            param_limits=param_limits
        )
        _log.setLevel(_prev_level)
        return g.fig
    
    def posterior_1d_overlays(self, param_name):
        if param_name not in self.results_list[0].param_names:
            raise ValueError(f"{param_name} not in parameter list.")
        
        plt.figure()
        g = plots.get_subplot_plotter()
        _log = logging.getLogger()
        _prev_level = _log.level
        _log.setLevel(logging.ERROR)
        g.plot_1d(self.samples_list, param_name, legend_labels=self.labels)
        _log.setLevel(_prev_level)

        fig = plt.gcf()

        return fig