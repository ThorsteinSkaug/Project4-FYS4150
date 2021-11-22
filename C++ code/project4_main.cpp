#include <armadillo>
#include <iostream>
#include <random>

using namespace std;


/*
A function for initilizing a lattice of size L
Input varaibles:
- int L: The size of the lattice
- arma::mat lattice: The lattice to be initilized
- bool ordered: If ordered = true then all entries are given 1, else the lattice is initialized randomly
Output:
- Update the matrix lattice
*/
void init_lattice(int L, arma::mat& lattice, bool ordered){
  int N = L*L;
  if(ordered){
    lattice.fill(1); //Fill with 1's
  }
  else{
    //Random generator for a uniform distribution
    random_device rd;
    mt19937_64 generator(rd());
    generator.seed(74);
    uniform_real_distribution<double> distribution(0.0, 1.0);

    //Fill lattice with random values
    for(int i=0;i<L;i++){
      for(int j=0;j<L;j++){
        if(distribution(generator) < 0.5){
          lattice(i,j) = -1;
        }
        else{
          lattice(i,j) = 1;
        }
      }
    }
  }

}



/*
A function for calculating the energy a given lattice
Input varaibles:
- int L: The size of the lattice
- arma::mat lattice: The lattice we want to calculate energy for
Output:
- The energy of the given lattice
*/
double energy(int L, arma::mat lattice){
  double E = 0;
  for(int i=0;i<L;i++){
    for(int j=0;j<L;j++){
      //Calculate the sum of the neighbour to the right and down for all entries in the lattice
      E -= lattice(i,j)*(lattice(i,(j+1)%L)+lattice((j+1)%L,i));
    }
  }
  if(L == 2){
    E = E/2; //We double count if L=2, hence we need to divide by 2 if that is the case
  }
  return E;
}


/*
A function for calculating the magnetization a given lattice
Input varaibles:
- int L: The size of the lattice
- arma::mat lattice: The lattice we want to calculate magnetization for
Output:
- The magnetization of the given lattice
*/
double magnetization(int L, arma::mat lattice){
  double M = 0;
  for(int i=0;i<L;i++){
    for(int j=0;j<L;j++){
      //Sum every entery in the lattice
      M += lattice(i,j);
    }
  }
  return M;
}


/*
A function for calculating the specific heat capacity
Input varaibles:
- int L: The size of the lattice
- double T: The temperature
- double EE: The sum of all energies after n_cycles cycles
- double EE2: The sum of all energies squared after n_cycles cycles
- int n_cycles: The number of cycles used to calculate EE and EE2
Output:
- The specific heat capacity
*/
double C_v(int L, double T, double EE, double EE2, int n_cycles){
  int N = L*L;
  EE = EE/n_cycles; //Find estimate of expected epsilon
  EE2 = EE2/n_cycles; //Find estimate of expected epsilon^2
  return (1/(N*T*T))*(EE2-EE*EE);
}


/*
A function for calculating the susceptibility
Input varaibles:
- int L: The size of the lattice
- double T: The temperature
- double EM: The sum of all magnetization after n_cycles cycles
- double EM2: The sum of all magnetization squared after n_cycles cycles
- int n_cycles: The number of cycles used to calculate EM and EM2
Output:
- The susceptibility
*/
double chi(int L, double T, double EM, double EM2, int n_cycles){
  int N = L*L;
  EM = EM/n_cycles; //Find estimate of expected m
  EM2 = EM2/n_cycles; //Find estimate of expected m^2
  return (1/(N*T))*(EM2-EM*EM);
}



/*
A function for calculating all of the boltzmann factors.
Input varaibles:
- double T: The temperature
- arma::vec exp_l: A vector of size (at least) 17
Output:
- exp_l with all the boltzmann factors at index {0, 4, 8, 12, 16}, from highest to lowest.
*/
void make_exp_l(double T, arma::vec& exp_l){
  for(int i=0; i<17; i++){
    if(i%4==0){
      exp_l(i) = exp(-1/T*(i-8)); //Fill in places {0, 4, 8, 12, 16}
    }
    else{
      exp_l(i) = 0; //Fill the rest with 0's
    }
  }
}


/*
A function for doing the metropolis algorithm. One run of this algorithm is one MCMC cycle
Input varaibles:
- int L: The size of the lattice
- arma::mat lattice: The lattice we start with
Output:
- An updated lattice
- An updated value for M
- An updated value for E
*/
void metropolis_algo(int L, arma::mat& lattice, arma::vec exp_l, double& M, double& E){

  //Random generator for a uniform distribution
  random_device rd;
  mt19937_64 generator(rd());
  uniform_real_distribution<double> distribution(0.0, 1.0);

  int N = L*L;
  for(int i=0; i<N; i++){

      //Find random position
      int x = distribution(generator) * L;
      int y = distribution(generator) * L;

      //Find change in energy by flipping that position
      int dE = 2*lattice(x,y)*(lattice((L+x-1)%L,y) + lattice((x+1)%L,y) + lattice(x,(L+y-1)%L) + lattice(x,(y+1)%L));

      if(L==2){
        dE = dE/2; //If L=2 divide by 2 (for not to double count)
      }

      int idx_dE = dE + 8; //Find right Boltzmann factor
      double r = distribution(generator); //Find random value between 0 and 1

      if(r < exp_l(idx_dE)){ //If random value is less then Boltzmann factor -> Update lattice
        E += dE;
        M -= 2*lattice(x,y);
        lattice(x,y) *= -1;
      }
  }
}


/*
A function for doing Markov Chain Monte Carlo
Input varaibles:
- int L: The size of the lattice
- int n_cycles: The number of cycles to run
- double EE: A variable to store the sum of all energies after n_cycles cycles
- double EM: A variable to store the sum of all magnetization after n_cycles cycles
- double EE2: A variable to store the sum of all energies^2 after n_cycles cycles
- double EM2: A variable to store the sum of all magnetization^2 after n_cycles cycles
- arma::mat lattice: The lattice we start with
- double e: A variable for storing the energy at the final lattice
Output:
- An updated lattice
- An updated value for EE, EM, EE2, EM2, and e
*/
void MCMC(int L, double T, int n_cycles,  double& EE, double& EM, double& EE2, double& EM2, arma::mat& lattice, double& e){
  //Make and fill Boltzmann vector
  arma::vec exp_l(17);
  make_exp_l(T, exp_l);

  int N = L*L;

  //Initial values
  double E = energy(L, lattice);
  double M = magnetization(L, lattice);

  for(int i=0; i<n_cycles; i++){
    //Run metropolis algorithm and update values
    metropolis_algo(L, lattice, exp_l, M, E);
    EE = (E+EE);
    EE2 = E*E + EE2;
    EM = abs(M) + EM;
    EM2 = (M*M + EM2);
    e = E/N;
  }
}


/*
A function for writing to file when we want the values and number of cycles
Input varaibles:
- int n_cycles: The total number of cycles we want to run
- int when_to_write: When (After how many cycles) we want to write the values to the files
- double T: The temperature
- int L: The size of the lattice
- bool ord: If ordered = true then all entries are given 1, else the lattice is initialized randomly
Output:
- A file with the name "data_T="T"_"ord"_"L".txt"
*/
void write_file(int n_cycles, int when_to_write, double T, int L, bool ord){
  //Open file
  std::ofstream myfile;
  myfile.open("data_T="+std::to_string(T)+"_"+std::to_string(ord)+"_"+std::to_string(L)+".txt");

  //Make initial lattice
  arma::mat lattice(L,L);
  init_lattice(L, lattice, ord);

  //Initialize variables
  double EE=0., EM=0., EE2=0., EM2=0., e=0.;
  int N = L*L;

  //Write initial values to file
  myfile << std::scientific << 1 << " " << std::scientific << energy(L, lattice)/N << " " << std::scientific << magnetization(L, lattice)/N << " " << std::scientific << C_v(L, T, energy(L, lattice)/N, pow(energy(L, lattice)/N,2),1) << " " << std::scientific << chi(L, T, magnetization(L, lattice)/N, pow(magnetization(L, lattice)/N,2), 1) << '\n';
  for(int i = when_to_write; i<n_cycles+1; i += when_to_write){
    if(i % 100000 == 0){
      cout << i << "\n";
    }
      MCMC(L, T, when_to_write, EE, EM, EE2, EM2, lattice, e); //Run the MCMC code when_to_write cycles

      //Calculate values to write to file
      double C_v_T = C_v(L, T, EE, EE2, i);
      double chi_T = chi(L, T, EM, EM2, i);

      //Write values to file
      myfile << std::scientific << (i) << " " << std::scientific << EE/N/(i) << " " << std::scientific << EM/N/(i) << " " << std::scientific << C_v_T << " " << std::scientific << chi_T << '\n';
  }
}



/*
A function for writing to file when we want to sample epsilon values
Input varaibles:
- int n_times: The total number of sample we want
- int burn_in_time: The burn-in time before we start to sample
- double T: The temperature
- int L: The size of the lattice
- bool ord: If ordered = true then all entries are given 1, else the lattice is initialized randomly
Output:
- A file with the name "data_T="T"_"ord"_e_values.txt"
*/
void write_file_e(int n_times, int burn_in_time, double T, int L, bool ord){
  //Open file
  std::ofstream myfile;
  myfile.open("data_T="+std::to_string(T)+"_"+std::to_string(ord)+"_e_values"+".txt");

  //Initialize lattice
  arma::mat lattice(L,L);
  init_lattice(L, lattice, ord);

  //Initialize variables
  double EE=0., EM=0., EE2=0., EM2=0., e=0.;

  //Run burn-in time
  MCMC(L, T, burn_in_time, EE, EM, EE2, EM2, lattice, e);

  for(int i = 1; i<n_times; i ++){
    if(i % 100000 == 0){
      cout << i << "\n";
    }
      //For each cycle write e to file
      MCMC(L, T, 1, EE, EM, EE2, EM2, lattice, e);
      myfile << std::scientific << i << " " << std::scientific << e << '\n';
  }
}


int main(){
  //Initialize some variables
  int T = 1;
  double T2 = 2.4;
  bool ord_t = true;
  bool ord_f = false;
  int L = 2;
  int n_cycles = 250000;
  double T_val [2] = {1, 2.4};
  bool ord_l [2] = {true, false};

  //Exercise 4
  #pragma omp parallel for
  for(int i = 0; i<4; i++){
    if(i>1){
      write_file(n_cycles, 100, T_val[i-2], L, ord_t); //Run for ordered=true
    }
    else{
      write_file(n_cycles, 100, T_val[i], L, ord_f); //Run for ordered=false
    }
  }


  //Exercise 5
  int L2 = 20;
  #pragma omp parallel for
  for(int i = 0; i<4; i++){
    if(i>1){
      write_file(n_cycles, 100, T_val[i-2], L2, ord_t); //Run for ordered=true
    }
    else{
      write_file(n_cycles, 100, T_val[i], L2, ord_f); //Run for ordered=false
    }
  }

  //Exercise 6

  int burn_time = 25000; //Burn-in time as found in exercise 5

  //Timing tests:
  //Test without parallelization
  auto t1 = std::chrono::high_resolution_clock::now();

  write_file_e(n_cycles, burn_time, T, L2, ord_f);
  write_file_e(n_cycles, burn_time, T2, L2, ord_f);

  auto t2 = std::chrono::high_resolution_clock::now();
  double duration_seconds = std::chrono::duration<double>(t2 - t1).count(); //Measure time
  std::cout << "Test without parallelization: " << duration_seconds << " seconds for task 6 \n";

  //Test with parallelization
  auto t3 = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int i=0; i<2; i++){
    write_file_e(n_cycles, burn_time, T_val[i], L2, ord_f);
  }

  auto t4 = std::chrono::high_resolution_clock::now();
  double duration_seconds2 = std::chrono::duration<double>(t4 - t3).count(); //Measure time
  std::cout << "Test with parallelization: " << duration_seconds2 << " seconds for task 6 \n";


  //Exercise 8

  //A scan with large T steps, to roughly identify the subregion of T_C
  std::vector<double> Tval;
  for(double tval = 2.1; tval<=2.4; tval+=0.01){
    Tval.push_back(tval);
  }

  int n_cycles_ex8 = 100000;
  int bit = 25000;
  for(int Lval = 40; Lval<120; Lval += 20){
    //Open file
    std::ofstream myfile;
    myfile.open("data_L="+std::to_string(Lval)+"_ex8"+".txt");

    int N = Lval*Lval;
    #pragma omp parallel for
    for(int t=0;t<Tval.size();t++){
      std::cout << Lval << " " << Tval[t] << "\n";
      arma::mat lattice(Lval,Lval);
      init_lattice(Lval, lattice, ord_f);
      double EE = 0., EM=0., EE2 = 0., EM2=0., e=0.;

      //Burn-in time
      MCMC(Lval, Tval[t], bit, EE, EM, EE2, EM2, lattice, e);

      //Sampling after burn_in time
      EE = 0., EM=0., EE2 = 0., EM2=0., e=0.;
      MCMC(Lval, Tval[t], n_cycles_ex8, EE, EM, EE2, EM2, lattice, e);

      //Calculate neccessary values
      double C_v_T = C_v(Lval, Tval[t], EE, EE2, n_cycles_ex8);
      double chi_T = chi(Lval, Tval[t], EM, EM2, n_cycles_ex8);

      //Write to file
      myfile << std::scientific << Tval[t] << " " << std::scientific << EE/N/n_cycles_ex8 << " " << std::scientific << EM/N/n_cycles_ex8 << " " << std::scientific << C_v_T << " " << std::scientific << chi_T << '\n';
    }
  }


  //A more fine search
  std::vector<double> Tval2;
  for(double tval = 2.23; tval<=2.33; tval+=0.001){
    Tval2.push_back(tval);
  }

  int n_cycles_ex8_2 = 300000;
  int bit2 = 50000;
  for(int Lval = 40; Lval<120; Lval += 20){
    //Open file
    std::ofstream myfile;
    myfile.open("data_L="+std::to_string(Lval)+"_ex8_thorough"+".txt");

    int N = Lval*Lval;
    #pragma omp parallel for
    for(int t=0;t<Tval2.size();t++){
      std::cout << Lval << " " << Tval2[t] << "\n";
      arma::mat lattice(Lval,Lval);
      init_lattice(Lval, lattice, ord_f);
      double EE = 0., EM=0., EE2 = 0., EM2=0., e=0.;

      //Burn-in time
      MCMC(Lval, Tval2[t], bit2, EE, EM, EE2, EM2, lattice, e);

      //Sampling after burn_in time
      EE = 0., EM=0., EE2 = 0., EM2=0., e=0.;
      MCMC(Lval, Tval2[t], n_cycles_ex8_2, EE, EM, EE2, EM2, lattice, e);

      //Calculate neccessary values
      double C_v_T = C_v(Lval, Tval2[t], EE, EE2, n_cycles_ex8_2);
      double chi_T = chi(Lval, Tval2[t], EM, EM2, n_cycles_ex8_2);

      //Write to file
      myfile << std::scientific << Tval2[t] << " " << std::scientific << EE/N/n_cycles_ex8_2 << " " << std::scientific << EM/N/n_cycles_ex8_2 << " " << std::scientific << C_v_T << " " << std::scientific << chi_T << '\n';
    }
  }
}
