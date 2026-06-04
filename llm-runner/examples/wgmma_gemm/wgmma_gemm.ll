; ModuleID = 'builtin.module'
source_filename = "wgmma_gemm"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_1 = addrspace(3) global [4096 x float] zeroinitializer, align 128
@__shared_mem_0 = addrspace(3) global [4096 x float] zeroinitializer, align 128
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() #0

define ptx_kernel void @wgmma_gemm(i32 %v0, i32 %v1, i32 %v2, float %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, float %v8, ptr %v9, i64 %v10) {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v7, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v10, 1
  br label %bb0
bb0:
  %v17 = phi i32 [ %v0, %entry ]
  %v18 = phi i32 [ %v1, %entry ]
  %v19 = phi i32 [ %v2, %entry ]
  %v20 = phi float [ %v3, %entry ]
  %v21 = phi { ptr, i64 } [ %v12, %entry ]
  %v22 = phi { ptr, i64 } [ %v14, %entry ]
  %v23 = phi float [ %v8, %entry ]
  %v24 = phi { ptr, i64 } [ %v16, %entry ]
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v18 to i64
  %v27 = zext i32 %v19 to i64
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v29 = zext i32 %v28 to i64
  %v30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb2
bb2:
  %v31 = zext i32 %v30 to i64
  %v32 = mul i64 %v31, 64
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v34 = zext i32 %v33 to i64
  %v35 = mul i64 %v34, 64
  %v36 = urem i64 %v29, 64
  %v37 = add i64 %v32, %v36
  %v38 = udiv i64 %v29, 64
  %v39 = add i64 %v35, %v38
  %v40 = icmp ult i64 %v37, %v25
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb23, label %bb4
bb4:
  %v42 = icmp ult i64 %v39, %v26
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb23, label %bb5
bb5:
  br label %bb6
bb6:
  %v44 = phi float [ 0.0, %bb5 ], [ %v81, %bb21 ]
  %v45 = phi i64 [ 0, %bb5 ], [ %v98, %bb21 ]
  %v46 = icmp ult i64 %v45, %v27
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb22, label %bb7
bb7:
  %v48 = udiv i64 %v29, 64
  %v49 = add i64 %v45, %v48
  %v50 = icmp ult i64 %v49, %v27
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb11, label %bb8
bb8:
  %v52 = mul i64 %v37, %v27
  %v53 = add i64 %v52, %v49
  %v54 = extractvalue { ptr, i64 } %v21, 1
  %v55 = icmp ult i64 %v53, %v54
  br i1 %v55, label %bb9, label %bb24
bb9:
  %v56 = extractvalue { ptr, i64 } %v21, 0
  %v57 = getelementptr inbounds float, ptr %v56, i64 %v53
  %v58 = load float, ptr %v57
  %v59 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v29
  br label %bb10
bb10:
  store float %v58, ptr addrspace(3) %v59
  br label %bb11
bb11:
  %v60 = urem i64 %v29, 64
  %v61 = add i64 %v45, %v60
  %v62 = icmp ult i64 %v61, %v27
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb17, label %bb12
bb12:
  %v64 = udiv i64 %v29, 64
  %v65 = add i64 %v35, %v64
  %v66 = icmp ult i64 %v65, %v26
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb16, label %bb13
bb13:
  %v68 = mul i64 %v61, %v26
  %v69 = add i64 %v68, %v65
  %v70 = extractvalue { ptr, i64 } %v22, 1
  %v71 = icmp ult i64 %v69, %v70
  br i1 %v71, label %bb14, label %bb25
bb14:
  %v72 = extractvalue { ptr, i64 } %v22, 0
  %v73 = getelementptr inbounds float, ptr %v72, i64 %v69
  %v74 = load float, ptr %v73
  %v75 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v29
  br label %bb15
bb15:
  store float %v74, ptr addrspace(3) %v75
  br label %bb16
bb16:
  br label %bb17
bb17:
  call void @llvm.nvvm.barrier0() #0
  br label %bb18
bb18:
  %v77 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v78 = addrspacecast ptr addrspace(3) %v77 to ptr
  %v79 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v80 = addrspacecast ptr addrspace(3) %v79 to ptr
  br label %bb19
bb19:
  %v81 = phi float [ %v44, %bb18 ], [ %v96, %bb20 ]
  %v82 = phi i64 [ 0, %bb18 ], [ %v97, %bb20 ]
  %v83 = icmp ult i64 %v82, 64
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb21, label %bb20
bb20:
  %v85 = sub i64 %v37, %v32
  %v86 = mul i64 %v85, 64
  %v87 = add i64 %v86, %v82
  %v88 = getelementptr inbounds float, ptr %v78, i64 %v87
  %v89 = load float, ptr %v88
  %v90 = mul i64 %v82, 64
  %v91 = sub i64 %v39, %v35
  %v92 = add i64 %v90, %v91
  %v93 = getelementptr inbounds float, ptr %v80, i64 %v92
  %v94 = load float, ptr %v93
  %v95 = fmul float %v89, %v94
  %v96 = fadd float %v81, %v95
  %v97 = add i64 %v82, 1
  br label %bb19
bb21:
  %v98 = add i64 %v45, 64
  br label %bb6
bb22:
  %v99 = mul i64 %v37, %v26
  %v100 = add i64 %v99, %v39
  %v101 = extractvalue { ptr, i64 } %v24, 0
  %v102 = fmul float %v20, %v44
  %v103 = getelementptr inbounds float, ptr %v101, i64 %v100
  %v104 = load float, ptr %v103
  %v105 = fmul float %v23, %v104
  %v106 = fadd float %v102, %v105
  store float %v106, ptr %v103
  br label %bb23
bb23:
  ret void
bb24:
  unreachable
bb25:
  unreachable
}


attributes #0 = { convergent }
