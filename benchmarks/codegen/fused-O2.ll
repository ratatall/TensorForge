; ModuleID = 'tensorforge'
source_filename = "tensorforge"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin24.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem: none)
define void @tensorforge_run(ptr nofree readonly captures(none) %inputs, ptr noalias nofree writeonly captures(none) %output, ptr noalias nofree readnone captures(none) %scratch) local_unnamed_addr #0 {
entry:
  %A = load ptr, ptr %inputs, align 8
  %0 = getelementptr i8, ptr %inputs, i64 8
  %B = load ptr, ptr %0, align 8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %1 = getelementptr [4 x i8], ptr %A, i64 %index
  %2 = getelementptr i8, ptr %1, i64 16
  %3 = getelementptr i8, ptr %1, i64 32
  %4 = getelementptr i8, ptr %1, i64 48
  %wide.load = load <4 x float>, ptr %1, align 4
  %wide.load1 = load <4 x float>, ptr %2, align 4
  %wide.load2 = load <4 x float>, ptr %3, align 4
  %wide.load3 = load <4 x float>, ptr %4, align 4
  %5 = fmul <4 x float> %wide.load, splat (float 2.000000e+00)
  %6 = fmul <4 x float> %wide.load1, splat (float 2.000000e+00)
  %7 = fmul <4 x float> %wide.load2, splat (float 2.000000e+00)
  %8 = fmul <4 x float> %wide.load3, splat (float 2.000000e+00)
  %9 = getelementptr [4 x i8], ptr %B, i64 %index
  %10 = getelementptr i8, ptr %9, i64 16
  %11 = getelementptr i8, ptr %9, i64 32
  %12 = getelementptr i8, ptr %9, i64 48
  %wide.load4 = load <4 x float>, ptr %9, align 4
  %wide.load5 = load <4 x float>, ptr %10, align 4
  %wide.load6 = load <4 x float>, ptr %11, align 4
  %wide.load7 = load <4 x float>, ptr %12, align 4
  %13 = fadd <4 x float> %5, %wide.load4
  %14 = fadd <4 x float> %6, %wide.load5
  %15 = fadd <4 x float> %7, %wide.load6
  %16 = fadd <4 x float> %8, %wide.load7
  %17 = fcmp ogt <4 x float> %13, zeroinitializer
  %18 = fcmp ogt <4 x float> %14, zeroinitializer
  %19 = fcmp ogt <4 x float> %15, zeroinitializer
  %20 = fcmp ogt <4 x float> %16, zeroinitializer
  %21 = select <4 x i1> %17, <4 x float> %13, <4 x float> zeroinitializer
  %22 = select <4 x i1> %18, <4 x float> %14, <4 x float> zeroinitializer
  %23 = select <4 x i1> %19, <4 x float> %15, <4 x float> zeroinitializer
  %24 = select <4 x i1> %20, <4 x float> %16, <4 x float> zeroinitializer
  %25 = getelementptr [4 x i8], ptr %output, i64 %index
  %26 = getelementptr i8, ptr %25, i64 16
  %27 = getelementptr i8, ptr %25, i64 32
  %28 = getelementptr i8, ptr %25, i64 48
  store <4 x float> %21, ptr %25, align 4
  store <4 x float> %22, ptr %26, align 4
  store <4 x float> %23, ptr %27, align 4
  store <4 x float> %24, ptr %28, align 4
  %index.next = add nuw i64 %index, 16
  %29 = icmp eq i64 %index.next, 262144
  br i1 %29, label %after, label %vector.body, !llvm.loop !0

after:                                            ; preds = %vector.body
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem: none) "target-cpu"="apple-m4" "target-features" }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.unroll.runtime.disable"}
