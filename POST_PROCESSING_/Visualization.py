"""Visualization — post-processing plots for COSMIX MCMC/nested-sampling results.

Provides four plot families via MCMCVisualization:

    trace()     — parameter value vs sample index for each free parameter.
    corner()    — triangle plot of the joint posterior (via GetDist).
    residual()  — (data − theory) / σ for each likelihood, vs redshift
                   or named parameter index.
    plot_H()    — best-fit H(z) curve with optional posterior band.
    plot_dL()   — best-fit d_L(z) curve with optional posterior band.
    plot_fsigma8() — best-fit fσ₈(z) curve with optional posterior band.
    plot_Eg()   — best-fit E_G(z) curve with optional posterior band.
"""
from CORE_.Pipeline import Pipeline
from POST_PROCESSING_.ResultsContainer import MCMCResults
from CORE_.RequirementResolver import single_requirement
import matplotlib.pyplot as plt
from getdist import MCSamples, plots
import warnings
import logging
import numpy as np

class MCMCVisualization:
    def __init__(self, pipeline, results: MCMCResults):
        self.pipeline = pipeline
        self.results = results

    def trace(self):
        chain = self.results.chain
        names = self.results.latex_names

        npar = chain.shape[1]
        fig, axes = plt.subplots(npar, 1, sharex=True, figsize=(8, 2*npar))

        for i, ax in enumerate(axes):
            ax.plot(chain[:, i], lw=0.5, color="mediumseagreen")
            ax.set_ylabel(names[i])

        axes[-1].set_xlabel("Sample Index")
        plt.tight_layout()
        
        return fig

    def corner(self):
        chain = np.asarray(self.results.chain)
        weights = self.results.weights  # None for MCMC; importance weights for nested sampling
        if chain.ndim != 2 or chain.shape[0] < 10:
            warnings.warn(
                f"[MCMCVisualization] Only {len(chain)} posterior samples available; "
                "skipping corner plot."
            )
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Insufficient samples ({len(chain)}) for corner plot",
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # GetDist cannot build a triangle plot when all parameters are effectively fixed.
        # For nested sampling use weighted std; for MCMC use plain std.
        if weights is not None:
            w = weights / weights.sum()
            wt_mean = np.average(chain, weights=w, axis=0)
            varying = np.sqrt(np.average((chain - wt_mean)**2, weights=w, axis=0)) > 0
        else:
            varying = np.std(chain, axis=0) > 0
        if np.count_nonzero(varying) < 1:
            warnings.warn("[MCMCVisualization] No varying parameters in chain; skipping corner plot.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No varying parameters for corner plot",
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        chain_plot   = chain[:, varying]
        names_plot   = [n for n, keep in zip(self.results.param_names, varying) if keep]
        labels_plot  = [n for n, keep in zip(self.results.latex_names, varying) if keep]
        weights_plot = weights  # None for MCMC → equal-weight; array for nested → weighted KDE

        try:
            _log = logging.getLogger()
            _prev_level = _log.level
            _log.setLevel(logging.ERROR)
            # Pass importance weights so GetDist's KDE respects the posterior measure.
            # Cobaya/PolyChord standard: MCSamples(weights=...) with all dead points.
            # fine_bins / fine_bins_2D use GetDist defaults (1024) — the previous values
            # of 150/50 produced a 50×50-pixel 2D KDE grid, which is the primary cause
            # of jagged contours independent of sample count.
            samples = MCSamples(samples=chain_plot, names=names_plot, labels=labels_plot,
                                weights=weights_plot)
            g = plots.get_subplot_plotter()
            g.triangle_plot(samples, filled=True, title_limit=1)
            _log.setLevel(_prev_level)
            fig = plt.gcf()
            return fig
        except Exception as e:
            _log.setLevel(_prev_level)
            warnings.warn(f"[MCMCVisualization] Corner plot failed: {e}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Corner plot unavailable for this chain",
                    ha='center', va='center', transform=ax.transAxes)
            return fig

    def residual(self, kind="bestfit", normalize=False):
        """
        Plot residuals for a given likelihood

        theta: array-like - Parameter vector (e.g. best fit or posterior mean)
        kind: string - label for plot annotation (e.g. "best fit", "mean")
        normalize: bool - If True, plot residuals are divided by observational error
        ----------------------------------------------------------------------------
        residual = data - theory; residual/sigma if normalized

        plot: residual vs redshift with a residual = 0 line and possibly error bars
        """
        if kind == "bestfit":
            theta = self.results.best_fit
        elif kind == "mean":
            theta = self.results.mean
        else:
            raise ValueError(f"Unknown kind: {kind}")

        model = self.pipeline.model
        requirements = self.pipeline.requirements
        likelihoods = self.pipeline.likelihoods
        theory = model.compute_theory(theta, requirements)  
        res_likelihoods = [L for L in likelihoods if getattr(L, "produce_residuals", False)]
        nlik = len(res_likelihoods)

        # Give panels with many named tick labels extra vertical space
        _panel_heights = []
        for L in res_likelihoods:
            nlabels = len(getattr(L, '_param_labels', []))
            _panel_heights.append(3.5 if nlabels > 8 else 2.0)
        fig_height = sum(_panel_heights)
        fig_width  = max(10, max(len(getattr(L, '_param_labels', [])) * 0.55
                                 for L in res_likelihoods))

        fig, axes = plt.subplots(
            nlik, 1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"height_ratios": _panel_heights}
        )
        if nlik == 1:
            axes = [axes]

        for ax, L in zip(axes, res_likelihoods):
            comp_dict = L.get_theory_components(theta, theory)
            for obs, (x, d_vec, th_vec, sigma) in comp_dict.items():
                res_vec = d_vec - th_vec
                if normalize:
                    if sigma is None:
                        raise RuntimeError("Normalization requested but not sigma provided.")
                    res_vec = res_vec / sigma

                ax.axhline(0, color='k', lw=0.5)
                ax.errorbar(x, res_vec, yerr=sigma, fmt='+', ecolor='crimson',
                            ms=3, color='k', capsize=1.2, mec='k', elinewidth=0.7)
                ax.set_ylabel(L.name)
                # Likelihoods with named scalar parameters (e.g. CompCMB, SDSSDR16)
                # store _param_labels; use numeric x-indices and rotate labels.
                if hasattr(L, '_param_labels'):
                    ax.set_xticks(x)
                    nlabels = len(L._param_labels)
                    rotation = 45 if nlabels > 6 else 0
                    ha       = 'right' if nlabels > 6 else 'center'
                    ax.set_xticklabels(L._param_labels, rotation=rotation,
                                       ha=ha, fontsize=7)
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(r"Redshift ($z$)")

        plt.tight_layout()

        return fig

    def plot_H(self, kind="bestfit", posterior_bands=False, nsamples=100):
        if kind == "bestfit":
            theta = self.results.best_fit
        elif kind == "mean":
            theta = self.results.mean
        else:
            raise ValueError(f"Unknown kind: {kind}")
        
        model = self.pipeline.model
        requirements = self.pipeline.requirements
        likelihoods = self.pipeline.likelihoods
        theory = model.compute_theory(theta, requirements)

        H_keys = [k for k in requirements if k.startswith("H")]
        if not H_keys:
            raise RuntimeError("No H(z) requirement in pipeline")
        zmax = max(np.max(requirements[k]["z"]) for k in H_keys)

        fig, ax = plt.subplots(figsize=(8,5))

        # best-fit curve
        z_plot = np.linspace(0, zmax, 1000)
        theory_plot = model.compute_theory(theta, single_requirement("H", z_plot))
        H_plot = theory_plot.get("H")["values"]
        ax.plot(z_plot, H_plot, color="steelblue", label="Best-fit model", lw=1.5)

        #posterior bands
        if posterior_bands:
            chain = self.results.chain
            idx = np.random.choice(len(chain), size=nsamples, replace=False)
            H_samples = []
            for i in idx:
                th_i = model.compute_theory(chain[i], single_requirement("H", z_plot))
                if not getattr(th_i, "invalid", False):
                    vals = th_i.get("H")["values"]
                    if np.all(np.isfinite(vals)):
                        H_samples.append(vals)
            H_samples = np.asarray(H_samples)
            lo68, hi68 = np.percentile(H_samples, [16,84], axis=0)
            lo95, hi95 = np.percentile(H_samples, [2.5, 97.5], axis=0)

            ax.fill_between(z_plot, lo68, hi68, color="C0", alpha=0.4, label=r"$1\sigma$")
            ax.fill_between(z_plot, lo95, hi95, color="C0", alpha=0.2, label=r"$2\sigma$")

        for L in likelihoods:
            if not hasattr(L, "get_theory_components"):
                continue

            data = L.get_theory_components(theta, theory)
            if data is None:
                continue
            if "H" not in data:
                continue

            z, H_obs, _, sigma = data["H"]
            ax.errorbar(z, H_obs, yerr=sigma, fmt='+', alpha=0.7, ecolor='dimgray', color='firebrick', ms=6, label=L.name, elinewidth=1.2)

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$H(z)$")
        ax.legend(loc='upper left', fontsize=12)

        return fig

    def plot_mu(self, kind="bestfit", posterior_bands=False, nsamples=100):
        if kind == "bestfit":
            theta = self.results.best_fit
        elif kind == "mean":
            theta = self.results.mean
        else:
            raise ValueError(f"Unknown kind: {kind}")
        
        model = self.pipeline.model
        requirements = self.pipeline.requirements
        likelihoods = self.pipeline.likelihoods
        theory = model.compute_theory(theta, requirements)

        # Identify SN likelihoods
        SN_likelihoods = [L for L in likelihoods if hasattr(L, "get_theory_components") 
                          and L.get_theory_components(theta, theory) is not None 
                          and "mu" in L.get_theory_components(theta, theory)]

        if not SN_likelihoods:
            raise RuntimeError("No SN likelihood found providing mu(z)")
        
        # Plotting range
        zmax = 0.0
        for L in SN_likelihoods:
            comp = L.get_theory_components(theta, theory)
            z_data, _, _, _ = comp["mu"]
            zmax = max(zmax, np.max(z_data))

        zmin = max(1e-4, np.min(z_data[z_data > 0]))
        z_plot = np.linspace(zmin, zmax, 1000)

        # Best-fit Curve
        theory_plot = model.compute_theory(theta, single_requirement("dL", z_plot))

        for L in SN_likelihoods:

            # Posterior bands
            if posterior_bands:
                chain = self.results.chain
                idx = np.random.choice(len(chain), size=nsamples, replace=False)
            
                mu_samples = []

                for i in idx:
                    theta_i = chain[i]
                    theory_i = model.compute_theory(theta_i, single_requirement("dL", z_plot))
                    if getattr(theory_i, "invalid", False):
                        continue

                    mu_i = L.get_theory_components(theta_i, theory_i, z_override=z_plot)["mu"][2]
                    if np.all(np.isfinite(mu_i)):
                        mu_samples.append(mu_i)

            mu_samples = np.asarray(mu_samples)
            if mu_samples.shape[0] < 10:
                warnings.warn("Too few valid posterior samples for bands.")

            lo68, hi68 = np.percentile(mu_samples, [16,84], axis=0)
            lo95, hi95 = np.percentile(mu_samples, [2.5, 97.5], axis=0)

            fig, ax = plt.subplots(figsize=(8,5))

            ax.fill_between(z_plot, lo95, hi95, color="C0", alpha=0.2, label=r"$2\sigma$", zorder=1)
            ax.fill_between(z_plot, lo68, hi68, color="C0", alpha=0.4, label=r"$1\sigma$", zorder=2)
            ax.plot(z_plot, lo68, color="black", lw=0.8)
            ax.plot(z_plot, hi68, color="black", lw=0.8)

        comp = L.get_theory_components(theta, theory_plot, z_override=z_plot)
        z, _, mu_plot, _ = comp["mu"]
        ax.plot(z_plot, mu_plot, color="firebrick", lw=1, zorder=3, label=f"{L.name} Best-fit")

        # Data points
        theory_data = model.compute_theory(theta, requirements)

        for L in SN_likelihoods:
            comp = L.get_theory_components(theta, theory_data)
            z, mu_obs, _, sigma = comp["mu"]

            ax.errorbar(z, mu_obs, yerr=sigma, fmt='o', mec='royalblue', zorder=4, alpha=0.55, 
                        color='royalblue', ms=1.5, ecolor='royalblue', label=L.name, elinewidth=0.6)

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\mu(z)$")
        ax.legend(loc='lower right', fontsize=12)

        return fig
    

    def plot_fs8(self, kind="bestfit", posterior_bands=False, nsamples=100):
        if kind == "bestfit":
            theta = self.results.best_fit
        elif kind == "mean":
            theta = self.results.mean
        else:
            raise ValueError(f"Unknown kind: {kind}")
        
        model = self.pipeline.model
        requirements = self.pipeline.requirements
        likelihoods = self.pipeline.likelihoods
        theory = model.compute_theory(theta, requirements)

        # Identify RSD likelihoods dynamically
        RSD_likelihoods = [L for L in likelihoods if hasattr(L, "get_theory_components") 
                          and L.get_theory_components(theta, theory) is not None 
                          and "fsigma8" in L.get_theory_components(theta, theory)]

        if not RSD_likelihoods:
            raise RuntimeError("No RSD likelihood found providing fsigma8(z)")
        
        # Plotting Range
        zmax = 0.0
        for L in RSD_likelihoods:
            comp = L.get_theory_components(theta, theory)
            z_data, _, _, _ = comp["fsigma8"]
            zmax = max(zmax, np.max(z_data))

        zmin = max(1e-4, np.min(z_data[z_data > 0]))
        z_plot = np.linspace(zmin, zmax + 0.2, 1000)

        # Best-fit Curve
        theory_plot = model.compute_theory(theta, single_requirement("fsigma8", z_plot))

        #fig, ax = plt.subplots(figsize=(8,5))

        for L in RSD_likelihoods:
            # Posterior Bands
            if posterior_bands:
                chain = self.results.chain
                idx = np.random.choice(len(chain), size=nsamples, replace=False)
            
                fs8_samples = []

                for i in idx:
                    theta_i = chain[i]
                    theory_i = model.compute_theory(theta_i, single_requirement("fsigma8", z_plot))
                    if getattr(theory_i, "invalid", False):
                        continue

                    fs8_i = L.get_theory_components(theta_i, theory_i, z_override=z_plot)["fsigma8"][2]
                    if fs8_i is not None and np.all(np.isfinite(fs8_i)):
                        fs8_samples.append(fs8_i)

            fs8_samples = np.asarray(fs8_samples)
            if fs8_samples.shape[0] < 10:
                warnings.warn("Too few valid posterior samples for fsigma8 bands.")
                
            lo68, hi68 = np.percentile(fs8_samples, [16,84], axis=0)
            lo95, hi95 = np.percentile(fs8_samples, [2.5, 97.5], axis=0)

            fig, ax = plt.subplots(figsize=(8,5))

            ax.fill_between(z_plot, lo95, hi95, color="C0", alpha=0.2, label=r"$2\sigma$", zorder=1)
            ax.fill_between(z_plot, lo68, hi68, color="C0", alpha=0.4, label=r"$1\sigma$", zorder=2)
            ax.plot(z_plot, lo68, color="black", lw=0.8, zorder=3)
            ax.plot(z_plot, hi68, color="black", lw=0.8, zorder=3)

        comp = L.get_theory_components(theta, theory_plot, z_override=z_plot)
        z, _, fs8_plot, _ = comp["fsigma8"]
        ax.plot(z_plot, fs8_plot, color="firebrick", lw=1, zorder=3, label=f"{L.name} Best-fit")
            
        # Data Points
        theory_data = model.compute_theory(theta, requirements)

        for L in RSD_likelihoods:
            comp = L.get_theory_components(theta, theory_data)
            z, fs8_obs, _, sigma = comp["fsigma8"]

            ax.errorbar(z, fs8_obs, yerr=sigma, fmt='o', mec='darkgreen', zorder=5, alpha=0.75, 
                        color='mediumseagreen', ms=5, ecolor='darkgreen', label=L.name, elinewidth=1.2)

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$f\sigma_8(z)$")
        ax.legend(loc='upper right', fontsize=12)

        return fig
