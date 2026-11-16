# Using the research notebook

Open [Waveform_Exploration.ipynb](../../Waveform_Exploration.ipynb), edit its single
configuration cell, restart and Run All. Follow the root [fresh-machine setup](../../README.md)
and select that project environment as the kernel. Notebook metadata uses generic Python 3;
no pre-existing environment or machine path is required.

1. **Configure:** scenario, basis source, evaluation mode, dimensions, local paths,
   display IDs, test budgets, and optional training/discovery settings are grouped together.
2. **Run overview:** compact active-settings table; full configuration/provenance collapsed.
3. **Channel:** physical/effective matrices, with dimensions and path draw in details.
4. **Waveform structure:** complex time/frequency bases, coupling groups, all-member
   spectra/means, individual examples and received-basis propagation.
5. **Overlay comparisons:** all selected 8 or N/2 bases, complex and energy/magnitude
   representations. REVIEW controls inclusion; complete index mappings remain available.
6. **Error rates:** coverage summary, figures per constellation, detailed counts,
   intervals, unresolved zero-error bounds and stopping flags in a collapsed table.
7. **Run record:** ignored local archive with full metadata, arrays and source hashes.

The same plot set is retained: default DD13figures, CFO5figures. Inapplicable DD
views do not render in CFO mode. Source notebook remains output-free in Git.
Historical notebooks live only under support/internal/notebooks/; they are not entry points.

[Scientific controls and supported modes](CONFIGURABLE.md) ·
[Reading the figures](FIGURE_GUIDE.md) · [Included inputs and provenance](ARTIFACTS.md)
