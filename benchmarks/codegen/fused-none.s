	.build_version macos, 15, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_tensorforge_run                ; -- Begin function tensorforge_run
	.p2align	2
_tensorforge_run:                       ; @tensorforge_run
	.cfi_startproc
; %bb.0:                                ; %entry
	mov	x8, #0                          ; =0x0
	ldp	x9, x10, [x0]
	movi.2d	v0, #0000000000000000
LBB0_1:                                 ; %loop
                                        ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x9, x8, lsl #2]
	fadd	s1, s1, s1
	ldr	s2, [x10, x8, lsl #2]
	fadd	s1, s1, s2
	fcmp	s1, #0.0
	fcsel	s1, s1, s0, gt
	str	s1, [x1, x8, lsl #2]
	add	x8, x8, #1
	cmp	x8, #64, lsl #12                ; =262144
	b.lo	LBB0_1
; %bb.2:                                ; %after
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
