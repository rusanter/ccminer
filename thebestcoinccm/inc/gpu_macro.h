#pragma once


#ifndef Nrow
#error Macro definition of Nrow is required
#endif
#ifndef Ncol
#error Macro definition of Ncol is required
#endif
#ifndef Tcost
#error Macro definition of Tcost is required
#endif


#define NAME_VAR__(name, r, c, t)  name##_r##r##_c##c##_t##t
#define NAME_VAR_(name, r, c, t)   NAME_VAR__(name, r, c, t) // This macro is required to expand Nrow, Ncol, Tcost macro names to values!
#define NAME_VAR(name)             NAME_VAR_(name, Nrow, Ncol, Tcost)

