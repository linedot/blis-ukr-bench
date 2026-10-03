#pragma once
// SMSTART/SMSTOP hoisting for BLIS builds whose SME micro-kernel leaves
// streaming mode to its caller (BLIS_SME_SM_AT / BLIS_SME_ZA_AT).  Shared by
// bench.cpp and the GOTO harness; a no-op everywhere else.

// For the messy armsme smstart/smstop hoisting experiments

#define BLIS_SME_CLOBBERS \
	"memory", \
	 "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", \
	 "v8", "v9","v10","v11","v12","v13","v14","v15", \
	"v16","v17","v18","v19","v20","v21","v22","v23", \
	"v24","v25","v26","v27","v28","v29","v30","v31"

#if ( BLIS_SME_SM_AT != 0 ) && ( BLIS_SME_ZA_AT != 0 )
  #pragma message("SMSTART/STOP completely hoisted")
  #define BENCH_SME_ENTER() __asm__ volatile ( "smstart"    ::: BLIS_SME_CLOBBERS )
  #define BENCH_SME_EXIT()  __asm__ volatile ( "smstop"     ::: BLIS_SME_CLOBBERS )
#elif BLIS_SME_SM_AT != 0
  #pragma message("SMSTART/STOP partially (sm) hoisted")
  #define BENCH_SME_ENTER() __asm__ volatile ( "smstart sm" ::: BLIS_SME_CLOBBERS )
  #define BENCH_SME_EXIT()  __asm__ volatile ( "smstop sm"  ::: BLIS_SME_CLOBBERS )
#elif BLIS_SME_ZA_AT != 0
  #pragma message("SMSTART/STOP partially (za) hoisted")
  #define BENCH_SME_ENTER() __asm__ volatile ( "smstart za" ::: BLIS_SME_CLOBBERS )
  #define BENCH_SME_EXIT()  __asm__ volatile ( "smstop za"  ::: BLIS_SME_CLOBBERS )
#else
  #if defined(__ARM_FEATURE_SME)
    #pragma message("SMSTART/STOP fully inside microkernel")
  #endif
  #define BENCH_SME_ENTER() do {} while ( 0 )
  #define BENCH_SME_EXIT()  do {} while ( 0 )
#endif
