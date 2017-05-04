/*

 */
/*
 * Modified to compare with boussinesq.cc
 *
 */

#if defined (DM_PARALLEL)
#include <mpi.h>
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include <hhg.h>

//Sphere Initialization is missing

int main(int argc, char *argv[])
{
   int np = 1, rk = 0;
#ifdef DM_PARALLEL
   MPI_Init(&argc, &argv);
   MPI_Comm_size (MPI_COMM_WORLD, &np);
   MPI_Comm_rank (MPI_COMM_WORLD, &rk);
#endif

   ParamHandler params(argc, argv);
   
   std::stringstream logfname;
   logfname << argv[0] << "-np" << np << "-p" << rk << ".txt";

   hhgLogger log;
   if (params.getBool("logfile", false)) {
     log.redirectToFile(argv[0], params.getBool("append", false));
   }
   if (params.getBool("singleoutput", true)) {
     log.singleOutput();
   }
   log.printInfo();

   // print basic info
   logAdd ("# ");
   for(int i=0; i<argc; i++)
      logAdd ("%s ", argv[i]);
   logAdd ("\n");
   logAdd ("# %s (%s) compiled with %s\n", __FILE__, GIT_HASH, CXXFLAGS);
   logAdd ("# np %d rk %d\n", np, rk);
   
   lvl_t minLevel = params.getInt("minlevel", 2),
         maxLevel = params.getInt("maxlevel", 5);
   lvl_t solveLevel = minLevel; //added new
   
   lvl_t vtkoutputLevel = params.getInt("vtkoutput.level", maxLevel);
   if(vtkoutputLevel > maxLevel) {
      logAdd("# [%d] warning: vtkoutput.level limited to maxlevel.\n", rk);
      vtkoutputLevel = maxLevel;
   }
   if(minLevel > vtkoutputLevel) {
      logAdd("# [%d] error: minlevel > vtkoutput.level\n", rk);
      exit(-1);
   }
   assert(2 <= solveLevel && solveLevel <= maxLevel);
   
   hhgLevelInformation &info = hhgLevelInformation::instance();
   info.initialize(maxLevel+1);
   
   hhgVolumeMesh mesh;
   logAdd("# [%d] Setting up mesh\n", rk);
   
   mesh.setCoarsestLevel(solveLevel);
   mesh.setFinestLevel(maxLevel);
   hhgDirectInterface interface(params.getString("meshfile").c_str());
   mesh.setGeometryInterface(&interface);
   mesh.initialize();
   
   hhgUniformRefiner refiner(mesh);
   refiner.initialize();
   mesh.setRefiner(&refiner);
   
   //Viscosity parameter input is missing

   logAdd ("# [%d] Setting up operators\n", rk);
  
   hhgGlobalOperator<hhgTetrahedronDiffusionOperator>
      diffopr(mesh, 0, minLevel, Constant);

    
   hhgGlobalOperator<hhgTetrahedronDiffusionPreconditionerOperator>
      diffprecondopr(mesh, 0, minLevel, Constant);
   hhgGlobalOperator<hhgTetrahedronLumpedMassOperator>
      lumpedmassopr(mesh, 0, minLevel, Constant);
    
  hhgGlobalOperator<hhgTetrahedronDivergenceTXOperator>
	divTxopr(mesh, 0, minLevel, Constant);
  hhgGlobalOperator<hhgTetrahedronDivergenceTYOperator>
	divTyopr(mesh, 0, minLevel, Constant);
  hhgGlobalOperator<hhgTetrahedronDivergenceTZOperator>
	divTzopr(mesh, 0, minLevel, Constant);

  hhgGlobalOperator<hhgTetrahedronDivergenceXOperator> 
	divxopr(mesh, 0, minLevel, Constant);
  hhgGlobalOperator<hhgTetrahedronDivergenceYOperator>
	divyopr(mesh, 0, minLevel, Constant);
  hhgGlobalOperator<hhgTetrahedronDivergenceZOperator>
	divzopr(mesh, 0, minLevel, Constant);
    
   hhgFiniteVolumeAdvectionOperator
      fvm_opr(mesh, 0, minLevel);

   std::vector<hhgOperator*> diffoprs(3, &diffopr);
   std::vector<hhgOperator*> diffprecondoprs(1, &diffprecondopr);
   std::vector<hhgOperator*> lumpedmassoprs(1, &lumpedmassopr);
    
  std::vector<hhgOperator*> divToprs, divoprs;
  divToprs.push_back(&divTxopr);
  divToprs.push_back(&divTyopr);
  divToprs.push_back(&divTzopr);

  divoprs.push_back(&divxopr);
  divoprs.push_back(&divyopr);
  divoprs.push_back(&divzopr);

   std::vector<hhgOperator*> allops(1, diffoprs[0]);
   allops.push_back(&diffprecondopr);
   allops.push_back(&lumpedmassopr);
   allops.push_back(&fvm_opr);
   extend(allops, divToprs);
   extend(allops, divoprs);
//   extend(allops, gradoprs);
  
   bool multilevel_stab = params.getBool("multilevel_stab", false);
   hhgOperator* stabopr = 0;
   if(!multilevel_stab) {
      stabopr = new hhgGlobalOperator<hhgTetrahedronPSPGOperator>(mesh, 0, minLevel,Constant);
      allops.push_back(stabopr);
   }
   
  // set up initialization functions
  hhgConstantVolumeFunction<double> zero(0), one(1);
  std::vector<hhgVolumeFunction<double> *> zero_v(3, &zero);
  
  std::vector<hhgVolumeFunction<double> *> uexact_v;
  uexact_v.push_back(new hhgStringVolumeFunction<double>(params.getString("solutionx").c_str()));
  uexact_v.push_back(new hhgStringVolumeFunction<double>(params.getString("solutiony").c_str()));
  uexact_v.push_back(new hhgStringVolumeFunction<double>(params.getString("solutionz").c_str()));
  
  hhgVolumeFunction<double> *pexact_ = new hhgStringVolumeFunction<double>(params.getString("pressure").c_str());
  
  std::vector<hhgVolumeFunction<double> *> forcing_v;
  forcing_v.push_back(new hhgStringVolumeFunction<double>(params.getString("forcingx").c_str()));
  forcing_v.push_back(new hhgStringVolumeFunction<double>(params.getString("forcingy").c_str()));
  forcing_v.push_back(new hhgStringVolumeFunction<double>(params.getString("forcingz").c_str()));
  
  hhgStringVolumeFunction<double> c0_func(params.getString("c0").c_str()); // initial concentration
  
  FunctionManager functions(mesh, solveLevel, minLevel, "all");
  
  logAdd ("# [%d] Setting up unknowns vector\n", rk);
  std::vector<hhgScalarVariable*>
  u = functions.create("u", "velocity;vtkout", uexact_v);
  hhgScalarVariable& p = functions.create("p", "pressure;vtkout", &zero);
  hhgScalarVariable& prhs = functions.create("prhs", "prhs", &zero);
  hhgScalarVariable& pres = functions.create("pres", "pres", &zero);
   
  logAdd ("# [%d] Setting up exact solution\n", rk);
  std::vector<hhgScalarVariable*>
   uexactv = functions.create("uexact", "uexact;vtkout", uexact_v);
  hhgScalarVariable& pexact = functions.create("pexact", "pexact;vtkout", pexact_);

  logAdd ("# [%d] Setting up right hand side\n", rk);
  std::vector<hhgScalarVariable*>
   gv = functions.create("g", "forcing", forcing_v);
  std::vector<hhgScalarVariable*>
   rhsv = functions.create("rhs", "rhs;vtkout", zero_v);
  
  logAdd ("# [%d] Setting up residual\n", rk);
  std::vector<hhgScalarVariable*>
    resv = functions.create("res", "res", zero_v);
   
   // coordinates
   hhgStringVolumeFunction<double>
      x_func("x"),
      y_func("y"),
      z_func("z");

   hhgScalarVariable&
      x = functions.create("x", "coords", &x_func),
      y = functions.create("y", "coords", &y_func),
      z = functions.create("z", "coords", &z_func);
   
   // set some coefficient and coords
   std::vector<hhgScalarVariable*> up(u);
   if(params.getBool("correct_fluxes", true)) {
      if(multilevel_stab)
         logAdd ("# [%d] warning: flux-correction not implemented for multilevel stabilization!\n", rk);
      else
         up.push_back(&p); // flux correction also needs pressure
   }
   fvm_opr.setVelocity(up);
   fvm_opr.setCoordinates(functions("coords"));

   // old and new concentration
   hhgScalarVariable&
      c0 = functions.create("c0", "", &c0_func),
      c1 = functions.create("c1", "vtkout", &c0_func);
   
   // temporary variables
   hhgScalarVariable&
      tmp0 = functions.create("tmp0", "tmp", &zero),
      tmp1 = functions.create("tmp1", "tmp", &one);
  
  logAdd ("# [%d] Setting up uniform refiner\n", rk);
  std::vector<hhgScalarVariable*> funcs = functions("all");
  refiner.refine(funcs, allops, solveLevel, maxLevel);
  logAdd ("# [%d] Number of nodes in working set %llu\n", rk,
          hhgMeshStatistics::countNodes( mesh, maxLevel ));
  
  cycleLog cyclelog(&mesh, maxLevel);
  
  cyclelog.addEucNorm("Res_Euc", resv, functions(""), true);
  cyclelog.addEucNorm("Err_Euc", u, uexactv, true);
  cyclelog.addEucNorm("Err_Euc", functions("pressure"), functions("pexact"), true);
  cyclelog.addInfNorm("Err_Inf", u, uexactv, true);
  cyclelog.addTime("Time[sec]");
  
  logAdd("# [%d]  %-3s  ", rk, "It.");
  cyclelog.writeCaption();
  logAdd("\n");
  
  // initialize all vectors
  for(std::size_t level=solveLevel; level <= maxLevel; ++level) {
      for(unsigned i = 0; i != funcs.size(); ++i) {
         funcs[i]->initialize(level, *funcs[i]->getInitFunction());
         funcs[i]->initializeDirichletBoundaries(level, *funcs[i]->getInitFunction());
      }
      for(unsigned i = 0; i != 3; ++i)
         hhgLinAlg::FiniteElements::setupLoad(*gv[i], level, forcing_v[i]);
  }

  
  logAdd ("# [%d] Solving the Stokes equation\n", rk);

  hhgLinAlg::SchurCGSolver solver( mesh, minLevel, maxLevel, lumpedmassoprs,
                                   multilevel_stab, true );
 
  hhgEmptySolver empty(mesh, minLevel);
  int nu_coarse = params.getInt("nu_coarse", 10);
  hhgPCGSolver coarse_solver(mesh, minLevel, diffoprs, nu_coarse, empty);
  
  hhgSmoother smoother = stringToSmoother(params.getString("smoother", "GS"));
  double smoother_args = params.getDouble("relaxation_factor", 1.0);
  int nu_pre = params.getInt("nu_pre", 3), nu_post = params.getInt("nu_post", 3);
  int inner_iterations = params.getInt("inner_iterations", 5);
  double ptol = params.getDouble("ptol", 1e-8);
  
    
  for(std::size_t run=0; run < params.getInt("outer_iterations", 10); run++) {
    cyclelog.beforeCycle();
      
    solver.solve(u, p, gv, rhsv, prhs, resv, 
		 lumpedmassoprs, diffoprs, divToprs, divoprs, stabopr,
		 &coarse_solver, nu_coarse, nu_pre, nu_post,
		 maxLevel, solveLevel, smoother, &smoother_args,
                 3, 1, inner_iterations); // clean up parameter mess!!!

      
    cyclelog.afterCycleTiming();
    
    logAdd("  [%d]  %3d  ", rk, run);
    logAdd("\n", rk, run);
    cyclelog.writeCaption();
    logAdd("\n", rk, run);
    cyclelog.afterCycle();
  }
  
  logAdd ("# [%d] Solving the transport equation\n", rk);
  
   hhgPrimitiveGroup groups = WorkingSet;
   if(params.getBool("neumann", true))
      groups = groups | Boundary;
   
   // compute timestep via CFL number: dt = ccfl*h/|u|
   double const ccfl = params.getDouble("ccfl", .5);
   // (1) get velocity magnitude
   hhgLinAlg::Basic::magVector(u, tmp0, maxLevel);
   double umag = hhgLinAlg::Basic::maxNorm(tmp0, maxLevel);
   // (2) invert velocity magnitude and multiply by mesh-size
   double hmin = mesh.hmin(maxLevel);
   std::cout << "hmin: " << hmin << std::endl;
   double dt = ccfl*hmin/umag;
   
////   std::string const vtkoutput = params.getString("vtkoutput", "none"); // commented by Iniyan Kalaimani and added the below line
   std::string const
           vtkoutput = params.getString("vtkoutput", "none"),
           vtkoutdir = params.getString("vtkoutput.directory", "output");
   int const vtkoutputEvery = params.getInt("vtkoutput.every", 1);
   int i = 0, j = 0;
   
   // time stepping loop
   double t = 0.0, tend = params.getDouble("tend", 1.0);
   while(t < tend + 1e-12) {
   
      // increment simulation time
      t += dt;
   
      if(rk == 0) logAdd("t = %.3e\n", t);
      
      // save solution from last timestep
      hhgLinAlg::Basic::copyVariable(c1, c0, maxLevel, WorkingSet | Boundary);
      
      // explicit Euler step, i.e., c1 = (I - dt*A(u,p))*c0
      // (1) tmp1 = A(u,p)*c0
      hhgLinAlg::Basic::applyOperator(c0, fvm_opr, tmp1, maxLevel, WorkingSet | Boundary, Replace);
      
      // (2) c1 = c0 - dt*tmp1
      hhgLinAlg::Basic::addVariables(1.0, c0, -dt, tmp1, c1, maxLevel, groups);
      hhgLinAlg::Basic::sync(c1, maxLevel);
      
      // do VTK output?
      if((vtkoutput == "all" && (j++ % vtkoutputEvery == 0))
      or (vtkoutput == "end" and t > tend)) {
      
         std::stringstream ss;
         ss << i++;

////         hhgVTKFileInterface vtkInterface("transport", ss.str().c_str(), "output"); // commented by Iniyan Kalaimani and added the below line
    hhgVTKFileInterface vtkInterface("transport", ss.str().c_str(), vtkoutdir.c_str());
         
         // restrict to output level (if necessary)
         for(int l = 0; l < maxLevel - vtkoutputLevel; ++l)
            hhgLinAlg::Multigrid::coarsen(functions("vtkout"), functions("vtkout"),
                                          allops, maxLevel - l, Injection, groups);
         
         vtkInterface.storeGeometricLevel(functions.findVectorized("vtkout"), vtkoutputLevel,
                                          mesh.getPrimitiveStore());
      }
   }
  
   c0.initialize(maxLevel, *c0.getInitFunction());
   std::cout.precision(8);
   std::cout << std::scientific << hhgLinAlg::Basic::maxNorm(c1, c0, maxLevel) << " ";
   hhgLinAlg::Basic::addVariables(1.0, c0, -1.0, c1, tmp0, maxLevel, WorkingSet | Boundary, Replace);
   hhgLinAlg::Basic::sync(tmp0, maxLevel);
   std::cout << std::scientific << hhgLinAlg::Basic::eucNorm(tmp0, maxLevel, WorkingSet | Boundary) << std::endl;
     
  for(int i = 0; i < uexact_v.size(); ++i)
    delete uexact_v[i];
  for(int i = 0; i < forcing_v.size(); ++i)
    delete forcing_v[i];
  delete pexact_;

  functions.free(); // in the destructor this segfaults for some reason
  
#if defined (DM_PARALLEL)
  MPI_Finalize();
#endif
}

