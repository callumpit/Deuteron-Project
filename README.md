# Deuteron-Project
Scientific Computing project completed term 1 of the final year of my BSc Physics degree at the University of Birmingham.

The aim of this project was to use (anti)proton and (anti)neutron event data generated using Pythia 8 (original data file too large to upload) to simulate deuteron and anti-deuteron production
at the Large Hadron Collider (LHC) by using the coalescence model and a more recent, empirically based, model proposed by Dal and Raklev in their 2015 paper titled "An
Alternative Formation Model for Antideuterons from Dark Matter". A consistently accurate model for anti-deuteron formation provides more accurate predictions for cosmic ray
spectra. This allows more definative conclusions to be made on whether an observed excess of anti-deuterons in cosmic rays are from dark matter anhilliations or not.

Both models were successfully implemented using Python in project_final.py. Generated data for deuterons and anti-deuterons by each model are given in seperate .dat files.
From the data generated, the Dal-Raklev model produced a slightly harder transverse momentum spectrum than the coalescence model which better matches empirical data 
collected by experiments such as ALICE and BABAR. Transverse momentum histograms for data generated by both models appear as expected and can be generated using plots.py file.
PS1.py and PS2.py are solutions to the first two problem sheets of the module, written by Dr Philip Ilten (module lead). These contain useful classes which were implemented 
in project_final.py .


