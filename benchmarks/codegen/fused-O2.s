	.build_version macos, 15, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_tensorforge_run                ; -- Begin function tensorforge_run
	.p2align	2
_tensorforge_run:                       ; @tensorforge_run
; %bb.0:                                ; %entry
	ldp	x8, x9, [x0]
	add	x8, x8, #32
	add	x9, x9, #32
	add	x10, x1, #32
	mov	w11, #262144                    ; =0x40000
LBB0_1:                                 ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x8, #-32]
	ldp	q2, q3, [x8], #64
	fadd.4s	v0, v0, v0
	fadd.4s	v1, v1, v1
	fadd.4s	v2, v2, v2
	fadd.4s	v3, v3, v3
	ldp	q4, q5, [x9, #-32]
	ldp	q6, q7, [x9], #64
	fadd.4s	v0, v0, v4
	fadd.4s	v1, v1, v5
	fadd.4s	v2, v2, v6
	fadd.4s	v3, v3, v7
	fcmgt.4s	v4, v0, #0.0
	fcmgt.4s	v5, v1, #0.0
	fcmgt.4s	v6, v2, #0.0
	fcmgt.4s	v7, v3, #0.0
	and.16b	v0, v4, v0
	and.16b	v1, v5, v1
	and.16b	v2, v6, v2
	and.16b	v3, v7, v3
	stp	q0, q1, [x10, #-32]
	stp	q2, q3, [x10], #64
	subs	x11, x11, #16
	b.ne	LBB0_1
; %bb.2:                                ; %after
	ret
                                        ; -- End function
.subsections_via_symbols
