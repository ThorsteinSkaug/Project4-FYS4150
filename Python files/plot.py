import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")



def read(filename):
    df = pd.read_csv(filename, sep=' ', header=None)
    n_cycles = df[0].to_numpy()
    Ee = df[1].to_numpy()
    Em = df[2].to_numpy()
    Cv = df[3].to_numpy()
    chi = df[4].to_numpy()
    return n_cycles, Ee, Em, Cv, chi

def read_ex6(filename):
    df = pd.read_csv(filename, sep=' ', header=None)
    n_cycles = df[0].to_numpy()
    e = df[1].to_numpy()
    return n_cycles, e

def read_ex8(filename):
    df = pd.read_csv(filename, sep=' ', header=None)
    df = df.sort_values(df.columns[0])
    t = df[0].to_numpy()
    Ee = df[1].to_numpy()
    Em = df[2].to_numpy()
    Cv = df[3].to_numpy()
    chi = df[4].to_numpy()
    return t, Ee, Em, Cv, chi



def plot(val1, val2, title = '', x_label = '', y_label = '', label = []):
    for i in range(len(val2)):
        if label == []:
            plt.plot(val1[i], val2[i])
        else:
            plt.plot(val1[i], val2[i],label=(label[i]))
    if len(label) > 0:
        plt.legend()
    plt.title(title,fontsize=20)
    plt.xlabel(x_label, fontsize=17)
    plt.ylabel(y_label, fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.tight_layout()

def plot_ex6(e, bin, title = '', x_label = '', y_label = ''):
    plt.hist(e, bins = bin, density = True)
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=17)
    plt.ylabel(y_label, fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{title}.pdf', dpi=900)
    plt.show()

def analytical(T, N):
    B = 1/T
    EE = 1/(np.exp(-4*B)+np.exp(4*B)+6)*(4*np.exp(-4*B)-4*np.exp(4*B))
    EE2 = 1/(2*np.exp(-4*B)+2*np.exp(4*B)+12)*(32*np.exp(-4*B)+32*np.exp(4*B))
    EM = 1/(2*np.exp(-4*B)+2*np.exp(4*B)+12)*(8*np.exp(4*B)+16)
    EM2 = 1/(2*np.exp(-4*B)+2*np.exp(4*B)+12)*(32*np.exp(4*B)+32)
    #Ee_a = 1/(np.exp(-4*B)+np.exp(4*B)+6)*(np.exp(-4*B)-np.exp(4*B))
    Ee_a = EE/4
    #Em_a = 1/(2*np.exp(-4*B)+2*np.exp(4*B)+12)*(2*np.exp(4*B)+4)
    Em_a = EM/4
    #Cv_a = 1/(N*T*T)*(1/(np.exp(-4*B)+np.exp(4*B)+6)*(16*np.exp(-4*B)+16*np.exp(4*B))
    Cv_a = 1/(N*T*T)*(EE2-EE*EE)
    #chi_a =
    chi_a = 1/(N*T)*(EM2-EM*EM)
    return Ee_a, Em_a, Cv_a, chi_a



#Plot exercise 4
T1_0 = read('data_T=1.000000_0_2.txt')
T1_1 = read('data_T=1.000000_1_2.txt')
n_cycles= T1_0[0]

T = np.linspace(1,1,len(n_cycles))
Ee_a, Em_a, Cv_a, chi_a = analytical(T,4)

plot([n_cycles, n_cycles, n_cycles], [T1_0[1], T1_1[1], Ee_a], title = r"$\langle \epsilon \rangle$ for L=2 and T=1 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\langle \epsilon \rangle$ [ $J$]", label = ["Unordered", "Ordered", "Analytical"])
plt.savefig('eL=2.pdf', dpi=900)
plt.show()
plot([n_cycles, n_cycles, n_cycles], [T1_0[2], T1_1[2], Em_a], title = r"$\langle m \rangle$ for L=2 and T=1 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\langle m \rangle$", label = ["Unordered", "Ordered", "Analytical"])
plt.savefig('mL=2.pdf', dpi=900)
plt.show()
plot([n_cycles, n_cycles, n_cycles], [T1_0[3], T1_1[3], Cv_a], title = r"$C_V$ for L=2 and T=1 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$C_V$ [$k_B$]", label = ["Unordered", "Ordered", "Analytical"])
plt.savefig('CVL=2.pdf', dpi=900)
plt.show()
plot([n_cycles, n_cycles, n_cycles], [T1_0[4], T1_1[4], chi_a], title = r"$\chi$ for L=2and T=1 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\chi$ [ $J^{-1}$]", label = ["Unordered", "Ordered", "Analytical"])
plt.savefig('chiL=2.pdf', dpi=900)
plt.show()

print(rf"Analytical result for $\langle \epsilon \rangle$ is: {Ee_a[0]}. And our estimate after {T1_0[0][-1]} MCMC cycles is: {T1_0[1][-1]}")
print(fr"Analytical result for $\langle m \rangle$ is: {Em_a[0]}. And our estimate after {T1_0[0][-1]} MCMC cycles is: {T1_0[2][-1]}")
print(fr"Analytical result for $C_V$ is: {Cv_a[0]}. And our estimate after {T1_0[0][-1]} MCMC cycles is: {T1_0[3][-1]}")
print(fr"Analytical result for $\chi$ is: {chi_a[0]}. And our estimate after {T1_0[0][-1]} MCMC cycles is: {T1_0[4][-1]}")




#Plot exercise 5
T2_0 = read('data_T=2.400000_0_20.txt')
T2_1 = read('data_T=2.400000_1_20.txt')
n_cycles2 = T2_0[0]
plot([n_cycles2, n_cycles2], [T2_0[1], T2_1[1]], title = r"$\langle \epsilon \rangle$ for L=20 with T=2.4 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\langle \epsilon \rangle$ [ $J$]", label = ["Unordered", "Ordered"])
plt.savefig('eL=20.pdf', dpi=900)
plt.show()
plot([n_cycles2, n_cycles2], [T2_0[2], T2_1[2]], title = r"$\langle |m| \rangle$ for L=20 with T=2.4 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\langle |m| \rangle$", label = ["Unordered", "Ordered"])
plt.savefig('mL=20T=2.4.pdf', dpi=900)
plt.show()

T3_0 = read('data_T=1.000000_0_20.txt')
T3_1 = read('data_T=1.000000_1_20.txt')
n_cycles3 = T3_0[0]
plot([n_cycles3, n_cycles3], [T3_0[1], T3_1[1]], title = r"$\langle \epsilon \rangle$ for L=20 with T=1 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\langle \epsilon \rangle$ [ $J$]", label = ["Unordered", "Ordered"])
plt.savefig('eL=20T=1.pdf', dpi=900)
plt.show()
plot([n_cycles3, n_cycles3], [T3_0[2], T3_1[2]], title = r"$\langle |m| \rangle$ for L=20 with T=1 $J/k_B$", x_label = "Number of MCMC cycles", y_label = r"$\langle |m| \rangle$", label = ["Unordered", "Ordered"])
plt.savefig('mL=20T=1.pdf', dpi=900)
plt.show()



#Plot exercise 6
ex6_1 = read_ex6("data_T=1.000000_0_e_values.txt")
ex6_2 = read_ex6("data_T=2.400000_0_e_values.txt")
bin = int(max(len(np.unique(ex6_2[1])), len(np.unique(ex6_1[1])))/5)

plot_ex6(ex6_1[1], bin, title = "Normalized distribution for T=1 and L=20", x_label = r"$\epsilon$ [ $J$]", y_label = "Normalized amount")

plot_ex6(ex6_2[1], bin, title = "Normalized distribution for T=2.4 and L=20", x_label = r"$\epsilon$ [ $J$]", y_label = "Normalized amount")


from scipy.signal import savgol_filter


#Plot exercise 8
ex8_40_r = read_ex8("data_L=40_ex8.txt")
ex8_60_r = read_ex8("data_L=60_ex8.txt")
ex8_80_r = read_ex8("data_L=80_ex8.txt")
ex8_100_r = read_ex8("data_L=100_ex8.txt")

plot([ex8_40_r[0], ex8_60_r[0], ex8_80_r[0], ex8_100_r[0]], [ex8_40_r[4], ex8_60_r[4], ex8_80_r[4], ex8_100_r[4]], title = r"$\chi$ as a function of T", x_label = "T [ $J/k_B$]", y_label = r"$\chi$ [$J^{-1}$]", label = ["L=40", "L=60", "L=80", "L=100"])
plt.savefig('chiL=406080100_r.pdf', dpi=900)
plt.show()
plot([ex8_40_r[0], ex8_60_r[0], ex8_80_r[0], ex8_100_r[0]], [ex8_40_r[3], ex8_60_r[3], ex8_80_r[3], ex8_100_r[3]], title = r"$C_V$ as a function of T", x_label = r"T [ $J/k_B$]", y_label = r"$C_V$ [$k_B$]", label = ["L=40", "L=60", "L=80", "L=100"])
plt.savefig('CVL=406080100_r.pdf', dpi=900)
plt.show()



ex8_40_f = read_ex8("data_L=40_ex8_thorough.txt")
ex8_60_f = read_ex8("data_L=60_ex8_thorough.txt")
ex8_80_f = read_ex8("data_L=80_ex8_thorough.txt")
ex8_100_f = read_ex8("data_L=100_ex8_thorough.txt")


def crit(dat):
    fit_m = savgol_filter(dat[2],81,2)
    fit_cv = savgol_filter(dat[3],81,2)
    fit_chi = savgol_filter(dat[4],81,2)

    Tc_100_idx = np.argwhere(fit_cv == np.max(fit_cv))
    Tc_100 = dat[0][Tc_100_idx][0][0]

    return Tc_100, fit_m, fit_cv, fit_chi

Tc_40 , fit_m_40, fit_cv_40, fit_chi_40 =  crit(ex8_40_f)
Tc_60 , fit_m_60, fit_cv_60, fit_chi_60 =  crit(ex8_60_f)
Tc_80 , fit_m_80, fit_cv_80, fit_chi_80 =  crit(ex8_80_f)
Tc_100 , fit_m_100, fit_cv_100, fit_chi_100 =  crit(ex8_100_f)

plt.plot(ex8_40_f[0], ex8_40_f[3], 'o', label = "L=40")
plot([ex8_40_f[0]],  [fit_cv_40])
plt.plot(ex8_60_f[0], ex8_60_f[3], 'o', label = "L=60")
plot([ex8_60_f[0]],  [fit_cv_60])
plt.plot(ex8_80_f[0], ex8_80_f[3], 'o', label = "L=80")
plot([ex8_80_f[0]],  [fit_cv_80])
plt.plot(ex8_100_f[0], ex8_100_f[3], 'o', label = "L=100")
plot([ex8_100_f[0]],  [fit_cv_100])
plt.title(r"$C_V$ after thorough search", fontsize=20)
plt.xlabel(r"T [$J/k_B$]", fontsize=17)
plt.ylabel(r"$C_V$ [$k_B$]", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('thorough_search_cv.pdf', dpi=900)
plt.show()

plt.plot(ex8_40_f[0], ex8_40_f[4], 'o', label = "L=40")
plot([ex8_40_f[0]],  [fit_chi_40])
plt.plot(ex8_60_f[0], ex8_60_f[4], 'o', label = "L=60")
plot([ex8_60_f[0]],  [fit_chi_60])
plt.plot(ex8_80_f[0], ex8_80_f[4], 'o', label = "L=80")
plot([ex8_80_f[0]],  [fit_chi_80])
plt.plot(ex8_100_f[0], ex8_100_f[4], 'o', label = "L=100")
plot([ex8_100_f[0]],  [fit_chi_100])
plt.title(r"$\chi$ after thorough search", fontsize=20)
plt.xlabel(r"T [$J/k_B$]", fontsize=17)
plt.ylabel(r"$\chi$ [$k_B$]", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('thorough_search_chi.pdf', dpi=900)
plt.show()

plt.plot(ex8_100_f[0], ex8_100_f[2], 'o')
plot([ex8_100_f[0]],  [fit_m_100])
plt.title(r"$\langle |m| \rangle$ after thorough search for L=100", fontsize=20)
plt.xlabel(r"T [$J/k_B$]", fontsize=17)
plt.ylabel(r"$\langle |m| \rangle$", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('thorough_search_m.pdf', dpi=900)
plt.show()



#problem 9

from scipy.stats import linregress

Lval = np.array([1/40,1/60,1/80,1/100])
Tcval = np.array([Tc_40,Tc_60,Tc_80,Tc_100])

#Fit linear regression
fit = linregress(Lval, Tcval)

#Find line
line = [fit.intercept,fit.intercept + fit.slope* Lval[0]]

plt.tight_layout()
plt.plot(0, 2.269, 'o', color='red', label = "Analytical solution")
plt.plot(Lval, Tcval, 'o', label = "Estimated values")
plt.plot([0, 0.025], line, label = "Regression line")
plt.title(r"Linear regression of critical temperatures", fontsize=20)
plt.xlabel(r"1/L", fontsize=17)
plt.ylabel(r"$T_c$ [ $J/k_B$]", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.legend()
plt.savefig('lr_critical_T.pdf', dpi=900)
plt.show()

#Print our estimated value
print(f"This is our estimated value: {fit.intercept}")
