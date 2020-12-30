Intro
	Something about most data in functional genomics being observational rather than perturbational
	A little weird because functional genomics means different things to different people:
		**David meaning: assaying stuff * say that we mean this (“static data”)
		Only perturbations: MPRAs, crispr stuff
	Connect to “MR gaining widespread traction in statistical genetics for obtaining unbiased causal estimates of exposures on outcomes given appropriate assumptions. We’re exploring whether by leveraging DL sequence-to-function models to …”
	How much to focus on MR vs. treat it as an abstraction?
	Try and write it geared towards people who’d be interested in the results more than the method.

Are methods learning something causal vs. the utility of the estimates? Show promise of showing models learn something causal but don’t show that the estimates are useful.
	Networks would be more useful for biologists.

	Say something about distinction between how MR is used in statistical genetics vs. how we use it here. In statistical genetics have a population of individuals and look at genotype to phenotype. Vs. for us it's coming from in-silico mutagenesis.

Methods
	Inherited biases
		Example of GC bias (bias towards sequences with more GC content) -> fake correlated pleiotropy (good example to discuss)
	How much to discuss which confounders we expect and which assumptions our method satisfies?
		At least a bit

	Look at Plos computational biology
Results
	Simulation
		For simulations, show variant effects (predicted vs. true) plots for exp / out (in supplement)
			Show that in-silico mutagenesis works rather than assume it like in other papers
		Could do bar graph for global CEs for simulation
		Scale width of the kernel by std error
		2D KDE jointplot with width of the kernel as standard deviation
		Comparison between calibrated and uncalibrated estimates using simulation
	DeepSEA
		What if we do TF->chromatin and chromatin->TF?
		Do 1 pioneer on other TFs
