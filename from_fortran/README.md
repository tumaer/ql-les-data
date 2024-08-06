## Params
job_id        isoturb               Job ID
reynolds      [2500., 2856.]        Reynolds number
faku          1.                    velocity factor "fak*u"
initial_data  [0-6]                 [0: white noise, 1: Kolmogorov, 2: CBC1, 3: CBC3, 4:Pao, 5: TGV, 6: Pao]
ntopleft      4                     ?
ntopright     0                     ? 
captau        [0.5, 1., 50.]        secondary filter differential form parameter
itsec         0                     secodary filter incremental form parameter
isgs_model    [1, 2, 3, 4]          [1: Smagorinsky, 2-4: no model]
alpha_prim    [0., -0.2]            LES alpha for primary filter
alpha_sec     [0., 0.2, 0.49]       LES alpha for secondary filter
beta_prim     [0., 1.25, 1.75,  2.] LES beta. Specify either alpha or beta
beta_sec      [0., 1.25, 1.5, 2.5]  LES beta for secondary filter
itype_prim    1                     Primary LES filter [1: Pade, 2: Gauss]
itype_sec     1                     Secondary LES filter [1: Pade, 2: Power-Gauss]
iord_decon    [-1, 5]               LES only
iord_sec      [-1, 6]               Secondary filter order [>0: specified, 0: unspecified, <0: defined from primary]
iord_primreg  -1                    unused
dir_string    LES_A/                unused

## Controls
Input-parameter for cturb.
job_id    : job identification (character*50)
taumax    : maximum time (in delta_1/u_inf)
itmax     : maximum number of time steps
ioutst    : output after each ioutst time steps
dtauout   : output after each dtauout time increment
tauout_0  : left border of time interval considered for output
tauout_1  : right border of time interval considered for output
cfl       : Courant-Friedrichs-Lewy number
cyber(1)  : T : restart                    |  F : new start
cyber(2)  : T : output controlled by time  |  F : ctrl by iterations
cyber(3)  : 
cyber(4)  : T : LES                        |  F : DNS
cyber(5)  : T : TG initial cond            |  F : spectrum initial cond
cyber(6)  : T : random noise on TG         |  F : no noise on TG
cyber(7)  : T : secondary filtering        |  F : no secondary filtering
cyber(8)  :
cyber(9)  :
cyber(10) :
cyber(11) : 
cyber(12) :
cyber(13) :
cyber(14) :
cyber(15) :
cyber(16) :
cyber(17) :
cyber(18) :
cyber(19) :
cyber(20) :
cyber(21) :
cyber(22) :
cyber(23) :
cyber(24) :
cyber(25) :

## Run
```bash
make clean && make
bash ../exec/runscript
```