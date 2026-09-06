; ModuleID = 'tensorforge'
source_filename = "tensorforge"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin24.5.0"

define void @tensorforge_run(ptr %inputs, ptr noalias %output, ptr noalias %scratch) #0 {
entry:
  %0 = getelementptr ptr, ptr %inputs, i64 0
  %A = load ptr, ptr %0, align 8
  %1 = getelementptr ptr, ptr %inputs, i64 1
  %B = load ptr, ptr %1, align 8
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %2 = getelementptr float, ptr %A, i64 %i
  %v0 = load float, ptr %2, align 4
  %multiply = fmul float %v0, 2.000000e+00
  %3 = getelementptr float, ptr %B, i64 %i
  %v1 = load float, ptr %3, align 4
  %add = fadd float %multiply, %v1
  %positive = fcmp ogt float %add, 0.000000e+00
  %relu = select i1 %positive, float %add, float 0.000000e+00
  %4 = getelementptr float, ptr %output, i64 %i
  store float %relu, ptr %4, align 4
  %next = add i64 %i, 1
  %5 = icmp ult i64 %next, 262144
  br i1 %5, label %loop, label %after

after:                                            ; preds = %loop
  ret void
}

attributes #0 = { "target-cpu"="apple-m4" "target-features" }
