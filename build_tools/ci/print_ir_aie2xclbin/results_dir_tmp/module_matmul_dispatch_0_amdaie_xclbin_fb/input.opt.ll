; ModuleID = '/proj/gdba/jamesn/workspace/iree-amd-aie/build_tools/ci/print_ir_aie2xclbin/results_dir_tmp/module_matmul_dispatch_0_amdaie_xclbin_fb/input.ll'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@buf0 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf1 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf2 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf3 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf4 = external global [1 x [1 x [8 x [8 x [4 x [4 x i32]]]]]]
@buf5 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf6 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf7 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf8 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf9 = external global [1 x [1 x [8 x [8 x [4 x [4 x i32]]]]]]
@buf10 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf11 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf12 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf13 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf14 = external global [1 x [1 x [8 x [8 x [4 x [4 x i32]]]]]]
@buf15 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf16 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf17 = external global [1 x [1 x [8 x [4 x [8 x [4 x i32]]]]]]
@buf18 = external global [1 x [1 x [4 x [8 x [4 x [8 x i32]]]]]]
@buf19 = external global [1 x [1 x [8 x [8 x [4 x [4 x i32]]]]]]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
define void @matmul_dispatch_0_matmul_64x64x64_i32(ptr %0, ptr %1, ptr %2) local_unnamed_addr #1 {
  %4 = ptrtoint ptr %0 to i64
  %5 = and i64 %4, 63
  %6 = icmp eq i64 %5, 0
  tail call void @llvm.assume(i1 %6)
  %7 = ptrtoint ptr %1 to i64
  %8 = and i64 %7, 63
  %9 = icmp eq i64 %8, 0
  tail call void @llvm.assume(i1 %9)
  %10 = ptrtoint ptr %2 to i64
  %11 = and i64 %10, 63
  %12 = icmp eq i64 %11, 0
  tail call void @llvm.assume(i1 %12)
  ret void
}

; Function Attrs: noreturn nounwind
define void @core_0_2() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %463, %0
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf4 to i64), i64 31), i64 0))
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4096) @buf4, i8 0, i64 4096, i1 false)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf3 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf2 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf4 to i64), i64 31), i64 0))
  br label %.preheader13

.preheader13:                                     ; preds = %1, %229
  %2 = phi i64 [ 0, %1 ], [ %230, %229 ]
  %3 = shl nuw nsw i64 %2, 5
  %4 = shl nuw nsw i64 %2, 4
  br label %.preheader10

.preheader10:                                     ; preds = %.preheader13, %226
  %5 = phi i64 [ 0, %.preheader13 ], [ %227, %226 ]
  %6 = shl nuw nsw i64 %5, 7
  %7 = add nuw nsw i64 %6, %4
  %8 = or i64 %6, 1
  %9 = add nuw nsw i64 %8, %4
  %10 = or i64 %6, 2
  %11 = add nuw nsw i64 %10, %4
  %12 = or i64 %6, 3
  %13 = add nuw nsw i64 %12, %4
  br label %.preheader7

.preheader7:                                      ; preds = %.preheader10, %223
  %14 = phi i64 [ 0, %.preheader10 ], [ %224, %223 ]
  %15 = shl nuw nsw i64 %14, 8
  %16 = add nuw nsw i64 %15, %3
  %17 = shl nuw nsw i64 %14, 5
  %18 = add nuw nsw i64 %6, %17
  %19 = getelementptr i32, ptr @buf2, i64 %18
  %20 = load i32, ptr %19, align 4
  %21 = or i64 %18, 4
  %22 = getelementptr i32, ptr @buf2, i64 %21
  %23 = load i32, ptr %22, align 4
  %24 = or i64 %18, 8
  %25 = getelementptr i32, ptr @buf2, i64 %24
  %26 = load i32, ptr %25, align 4
  %27 = or i64 %18, 12
  %28 = getelementptr i32, ptr @buf2, i64 %27
  %29 = load i32, ptr %28, align 4
  %30 = or i64 %18, 16
  %31 = getelementptr i32, ptr @buf2, i64 %30
  %32 = load i32, ptr %31, align 4
  %33 = or i64 %18, 20
  %34 = getelementptr i32, ptr @buf2, i64 %33
  %35 = load i32, ptr %34, align 4
  %36 = or i64 %18, 24
  %37 = getelementptr i32, ptr @buf2, i64 %36
  %38 = load i32, ptr %37, align 4
  %39 = or i64 %18, 28
  %40 = getelementptr i32, ptr @buf2, i64 %39
  %41 = load i32, ptr %40, align 4
  %42 = add nuw nsw i64 %8, %17
  %43 = getelementptr i32, ptr @buf2, i64 %42
  %44 = load i32, ptr %43, align 4
  %45 = or i64 %42, 4
  %46 = getelementptr i32, ptr @buf2, i64 %45
  %47 = load i32, ptr %46, align 4
  %48 = or i64 %42, 8
  %49 = getelementptr i32, ptr @buf2, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = or i64 %42, 12
  %52 = getelementptr i32, ptr @buf2, i64 %51
  %53 = load i32, ptr %52, align 4
  %54 = or i64 %42, 16
  %55 = getelementptr i32, ptr @buf2, i64 %54
  %56 = load i32, ptr %55, align 4
  %57 = or i64 %42, 20
  %58 = getelementptr i32, ptr @buf2, i64 %57
  %59 = load i32, ptr %58, align 4
  %60 = or i64 %42, 24
  %61 = getelementptr i32, ptr @buf2, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = or i64 %42, 28
  %64 = getelementptr i32, ptr @buf2, i64 %63
  %65 = load i32, ptr %64, align 4
  %66 = add nuw nsw i64 %10, %17
  %67 = getelementptr i32, ptr @buf2, i64 %66
  %68 = load i32, ptr %67, align 4
  %69 = or i64 %66, 4
  %70 = getelementptr i32, ptr @buf2, i64 %69
  %71 = load i32, ptr %70, align 4
  %72 = or i64 %66, 8
  %73 = getelementptr i32, ptr @buf2, i64 %72
  %74 = load i32, ptr %73, align 4
  %75 = or i64 %66, 12
  %76 = getelementptr i32, ptr @buf2, i64 %75
  %77 = load i32, ptr %76, align 4
  %78 = or i64 %66, 16
  %79 = getelementptr i32, ptr @buf2, i64 %78
  %80 = load i32, ptr %79, align 4
  %81 = or i64 %66, 20
  %82 = getelementptr i32, ptr @buf2, i64 %81
  %83 = load i32, ptr %82, align 4
  %84 = or i64 %66, 24
  %85 = getelementptr i32, ptr @buf2, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = or i64 %66, 28
  %88 = getelementptr i32, ptr @buf2, i64 %87
  %89 = load i32, ptr %88, align 4
  %90 = add nuw nsw i64 %12, %17
  %91 = getelementptr i32, ptr @buf2, i64 %90
  %92 = load i32, ptr %91, align 4
  %93 = or i64 %90, 4
  %94 = getelementptr i32, ptr @buf2, i64 %93
  %95 = load i32, ptr %94, align 4
  %96 = or i64 %90, 8
  %97 = getelementptr i32, ptr @buf2, i64 %96
  %98 = load i32, ptr %97, align 4
  %99 = or i64 %90, 12
  %100 = getelementptr i32, ptr @buf2, i64 %99
  %101 = load i32, ptr %100, align 4
  %102 = or i64 %90, 16
  %103 = getelementptr i32, ptr @buf2, i64 %102
  %104 = load i32, ptr %103, align 4
  %105 = or i64 %90, 20
  %106 = getelementptr i32, ptr @buf2, i64 %105
  %107 = load i32, ptr %106, align 4
  %108 = or i64 %90, 24
  %109 = getelementptr i32, ptr @buf2, i64 %108
  %110 = load i32, ptr %109, align 4
  %111 = or i64 %90, 28
  %112 = getelementptr i32, ptr @buf2, i64 %111
  %113 = load i32, ptr %112, align 4
  br label %.preheader5

.preheader5:                                      ; preds = %.preheader7, %.preheader5
  %114 = phi i64 [ 0, %.preheader7 ], [ %221, %.preheader5 ]
  %115 = shl nuw nsw i64 %114, 3
  %116 = add nuw nsw i64 %16, %115
  %117 = shl nuw nsw i64 %114, 2
  %118 = getelementptr i32, ptr @buf3, i64 %116
  %119 = add nuw nsw i64 %7, %117
  %120 = getelementptr i32, ptr @buf4, i64 %119
  %.promoted = load i32, ptr %120, align 4
  %121 = load i32, ptr %118, align 4
  %122 = mul i32 %20, %121
  %123 = add i32 %.promoted, %122
  %124 = or i64 %116, 1
  %125 = getelementptr i32, ptr @buf3, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = mul i32 %23, %126
  %128 = add i32 %123, %127
  %129 = or i64 %116, 2
  %130 = getelementptr i32, ptr @buf3, i64 %129
  %131 = load i32, ptr %130, align 4
  %132 = mul i32 %26, %131
  %133 = add i32 %128, %132
  %134 = or i64 %116, 3
  %135 = getelementptr i32, ptr @buf3, i64 %134
  %136 = load i32, ptr %135, align 4
  %137 = mul i32 %29, %136
  %138 = add i32 %133, %137
  %139 = or i64 %116, 4
  %140 = getelementptr i32, ptr @buf3, i64 %139
  %141 = load i32, ptr %140, align 4
  %142 = mul i32 %32, %141
  %143 = add i32 %138, %142
  %144 = or i64 %116, 5
  %145 = getelementptr i32, ptr @buf3, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %35, %146
  %148 = add i32 %143, %147
  %149 = or i64 %116, 6
  %150 = getelementptr i32, ptr @buf3, i64 %149
  %151 = load i32, ptr %150, align 4
  %152 = mul i32 %38, %151
  %153 = add i32 %148, %152
  %154 = or i64 %116, 7
  %155 = getelementptr i32, ptr @buf3, i64 %154
  %156 = load i32, ptr %155, align 4
  %157 = mul i32 %41, %156
  %158 = add i32 %153, %157
  store i32 %158, ptr %120, align 4
  %159 = add nuw nsw i64 %9, %117
  %160 = getelementptr i32, ptr @buf4, i64 %159
  %.promoted.1 = load i32, ptr %160, align 4
  %161 = mul i32 %44, %121
  %162 = add i32 %.promoted.1, %161
  %163 = mul i32 %47, %126
  %164 = add i32 %162, %163
  %165 = mul i32 %50, %131
  %166 = add i32 %164, %165
  %167 = mul i32 %53, %136
  %168 = add i32 %166, %167
  %169 = mul i32 %56, %141
  %170 = add i32 %168, %169
  %171 = mul i32 %59, %146
  %172 = add i32 %170, %171
  %173 = mul i32 %62, %151
  %174 = add i32 %172, %173
  %175 = mul i32 %65, %156
  %176 = add i32 %174, %175
  store i32 %176, ptr %160, align 4
  %177 = add nuw nsw i64 %11, %117
  %178 = getelementptr i32, ptr @buf4, i64 %177
  %.promoted.2 = load i32, ptr %178, align 4
  %179 = load i32, ptr %118, align 4
  %180 = mul i32 %68, %179
  %181 = add i32 %.promoted.2, %180
  %182 = load i32, ptr %125, align 4
  %183 = mul i32 %71, %182
  %184 = add i32 %181, %183
  %185 = load i32, ptr %130, align 4
  %186 = mul i32 %74, %185
  %187 = add i32 %184, %186
  %188 = load i32, ptr %135, align 4
  %189 = mul i32 %77, %188
  %190 = add i32 %187, %189
  %191 = load i32, ptr %140, align 4
  %192 = mul i32 %80, %191
  %193 = add i32 %190, %192
  %194 = load i32, ptr %145, align 4
  %195 = mul i32 %83, %194
  %196 = add i32 %193, %195
  %197 = load i32, ptr %150, align 4
  %198 = mul i32 %86, %197
  %199 = add i32 %196, %198
  %200 = load i32, ptr %155, align 4
  %201 = mul i32 %89, %200
  %202 = add i32 %199, %201
  store i32 %202, ptr %178, align 4
  %203 = add nuw nsw i64 %13, %117
  %204 = getelementptr i32, ptr @buf4, i64 %203
  %.promoted.3 = load i32, ptr %204, align 4
  %205 = mul i32 %92, %179
  %206 = add i32 %.promoted.3, %205
  %207 = mul i32 %95, %182
  %208 = add i32 %206, %207
  %209 = mul i32 %98, %185
  %210 = add i32 %208, %209
  %211 = mul i32 %101, %188
  %212 = add i32 %210, %211
  %213 = mul i32 %104, %191
  %214 = add i32 %212, %213
  %215 = mul i32 %107, %194
  %216 = add i32 %214, %215
  %217 = mul i32 %110, %197
  %218 = add i32 %216, %217
  %219 = mul i32 %113, %200
  %220 = add i32 %218, %219
  store i32 %220, ptr %204, align 4
  %221 = add nuw nsw i64 %114, 1
  %222 = icmp ult i64 %114, 3
  br i1 %222, label %.preheader5, label %223

223:                                              ; preds = %.preheader5
  %224 = add nuw nsw i64 %14, 1
  %225 = icmp ult i64 %14, 3
  br i1 %225, label %.preheader7, label %226

226:                                              ; preds = %223
  %227 = add nuw nsw i64 %5, 1
  %228 = icmp ult i64 %5, 7
  br i1 %228, label %.preheader10, label %229

229:                                              ; preds = %226
  %230 = add nuw nsw i64 %2, 1
  %231 = icmp ult i64 %2, 7
  br i1 %231, label %.preheader13, label %232

232:                                              ; preds = %229
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf1 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf0 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf4 to i64), i64 31), i64 0))
  br label %.preheader12

.preheader12:                                     ; preds = %232, %460
  %233 = phi i64 [ 0, %232 ], [ %461, %460 ]
  %234 = shl nuw nsw i64 %233, 5
  %235 = shl nuw nsw i64 %233, 4
  br label %.preheader9

.preheader9:                                      ; preds = %.preheader12, %457
  %236 = phi i64 [ 0, %.preheader12 ], [ %458, %457 ]
  %237 = shl nuw nsw i64 %236, 7
  %238 = add nuw nsw i64 %237, %235
  %239 = or i64 %237, 1
  %240 = add nuw nsw i64 %239, %235
  %241 = or i64 %237, 2
  %242 = add nuw nsw i64 %241, %235
  %243 = or i64 %237, 3
  %244 = add nuw nsw i64 %243, %235
  br label %.preheader6

.preheader6:                                      ; preds = %.preheader9, %454
  %245 = phi i64 [ 0, %.preheader9 ], [ %455, %454 ]
  %246 = shl nuw nsw i64 %245, 8
  %247 = add nuw nsw i64 %246, %234
  %248 = shl nuw nsw i64 %245, 5
  %249 = add nuw nsw i64 %237, %248
  %250 = getelementptr i32, ptr @buf0, i64 %249
  %251 = load i32, ptr %250, align 4
  %252 = or i64 %249, 4
  %253 = getelementptr i32, ptr @buf0, i64 %252
  %254 = load i32, ptr %253, align 4
  %255 = or i64 %249, 8
  %256 = getelementptr i32, ptr @buf0, i64 %255
  %257 = load i32, ptr %256, align 4
  %258 = or i64 %249, 12
  %259 = getelementptr i32, ptr @buf0, i64 %258
  %260 = load i32, ptr %259, align 4
  %261 = or i64 %249, 16
  %262 = getelementptr i32, ptr @buf0, i64 %261
  %263 = load i32, ptr %262, align 4
  %264 = or i64 %249, 20
  %265 = getelementptr i32, ptr @buf0, i64 %264
  %266 = load i32, ptr %265, align 4
  %267 = or i64 %249, 24
  %268 = getelementptr i32, ptr @buf0, i64 %267
  %269 = load i32, ptr %268, align 4
  %270 = or i64 %249, 28
  %271 = getelementptr i32, ptr @buf0, i64 %270
  %272 = load i32, ptr %271, align 4
  %273 = add nuw nsw i64 %239, %248
  %274 = getelementptr i32, ptr @buf0, i64 %273
  %275 = load i32, ptr %274, align 4
  %276 = or i64 %273, 4
  %277 = getelementptr i32, ptr @buf0, i64 %276
  %278 = load i32, ptr %277, align 4
  %279 = or i64 %273, 8
  %280 = getelementptr i32, ptr @buf0, i64 %279
  %281 = load i32, ptr %280, align 4
  %282 = or i64 %273, 12
  %283 = getelementptr i32, ptr @buf0, i64 %282
  %284 = load i32, ptr %283, align 4
  %285 = or i64 %273, 16
  %286 = getelementptr i32, ptr @buf0, i64 %285
  %287 = load i32, ptr %286, align 4
  %288 = or i64 %273, 20
  %289 = getelementptr i32, ptr @buf0, i64 %288
  %290 = load i32, ptr %289, align 4
  %291 = or i64 %273, 24
  %292 = getelementptr i32, ptr @buf0, i64 %291
  %293 = load i32, ptr %292, align 4
  %294 = or i64 %273, 28
  %295 = getelementptr i32, ptr @buf0, i64 %294
  %296 = load i32, ptr %295, align 4
  %297 = add nuw nsw i64 %241, %248
  %298 = getelementptr i32, ptr @buf0, i64 %297
  %299 = load i32, ptr %298, align 4
  %300 = or i64 %297, 4
  %301 = getelementptr i32, ptr @buf0, i64 %300
  %302 = load i32, ptr %301, align 4
  %303 = or i64 %297, 8
  %304 = getelementptr i32, ptr @buf0, i64 %303
  %305 = load i32, ptr %304, align 4
  %306 = or i64 %297, 12
  %307 = getelementptr i32, ptr @buf0, i64 %306
  %308 = load i32, ptr %307, align 4
  %309 = or i64 %297, 16
  %310 = getelementptr i32, ptr @buf0, i64 %309
  %311 = load i32, ptr %310, align 4
  %312 = or i64 %297, 20
  %313 = getelementptr i32, ptr @buf0, i64 %312
  %314 = load i32, ptr %313, align 4
  %315 = or i64 %297, 24
  %316 = getelementptr i32, ptr @buf0, i64 %315
  %317 = load i32, ptr %316, align 4
  %318 = or i64 %297, 28
  %319 = getelementptr i32, ptr @buf0, i64 %318
  %320 = load i32, ptr %319, align 4
  %321 = add nuw nsw i64 %243, %248
  %322 = getelementptr i32, ptr @buf0, i64 %321
  %323 = load i32, ptr %322, align 4
  %324 = or i64 %321, 4
  %325 = getelementptr i32, ptr @buf0, i64 %324
  %326 = load i32, ptr %325, align 4
  %327 = or i64 %321, 8
  %328 = getelementptr i32, ptr @buf0, i64 %327
  %329 = load i32, ptr %328, align 4
  %330 = or i64 %321, 12
  %331 = getelementptr i32, ptr @buf0, i64 %330
  %332 = load i32, ptr %331, align 4
  %333 = or i64 %321, 16
  %334 = getelementptr i32, ptr @buf0, i64 %333
  %335 = load i32, ptr %334, align 4
  %336 = or i64 %321, 20
  %337 = getelementptr i32, ptr @buf0, i64 %336
  %338 = load i32, ptr %337, align 4
  %339 = or i64 %321, 24
  %340 = getelementptr i32, ptr @buf0, i64 %339
  %341 = load i32, ptr %340, align 4
  %342 = or i64 %321, 28
  %343 = getelementptr i32, ptr @buf0, i64 %342
  %344 = load i32, ptr %343, align 4
  br label %.preheader4

.preheader4:                                      ; preds = %.preheader6, %.preheader4
  %345 = phi i64 [ 0, %.preheader6 ], [ %452, %.preheader4 ]
  %346 = shl nuw nsw i64 %345, 3
  %347 = add nuw nsw i64 %247, %346
  %348 = shl nuw nsw i64 %345, 2
  %349 = getelementptr i32, ptr @buf1, i64 %347
  %350 = add nuw nsw i64 %238, %348
  %351 = getelementptr i32, ptr @buf4, i64 %350
  %.promoted15 = load i32, ptr %351, align 4
  %352 = load i32, ptr %349, align 4
  %353 = mul i32 %251, %352
  %354 = add i32 %.promoted15, %353
  %355 = or i64 %347, 1
  %356 = getelementptr i32, ptr @buf1, i64 %355
  %357 = load i32, ptr %356, align 4
  %358 = mul i32 %254, %357
  %359 = add i32 %354, %358
  %360 = or i64 %347, 2
  %361 = getelementptr i32, ptr @buf1, i64 %360
  %362 = load i32, ptr %361, align 4
  %363 = mul i32 %257, %362
  %364 = add i32 %359, %363
  %365 = or i64 %347, 3
  %366 = getelementptr i32, ptr @buf1, i64 %365
  %367 = load i32, ptr %366, align 4
  %368 = mul i32 %260, %367
  %369 = add i32 %364, %368
  %370 = or i64 %347, 4
  %371 = getelementptr i32, ptr @buf1, i64 %370
  %372 = load i32, ptr %371, align 4
  %373 = mul i32 %263, %372
  %374 = add i32 %369, %373
  %375 = or i64 %347, 5
  %376 = getelementptr i32, ptr @buf1, i64 %375
  %377 = load i32, ptr %376, align 4
  %378 = mul i32 %266, %377
  %379 = add i32 %374, %378
  %380 = or i64 %347, 6
  %381 = getelementptr i32, ptr @buf1, i64 %380
  %382 = load i32, ptr %381, align 4
  %383 = mul i32 %269, %382
  %384 = add i32 %379, %383
  %385 = or i64 %347, 7
  %386 = getelementptr i32, ptr @buf1, i64 %385
  %387 = load i32, ptr %386, align 4
  %388 = mul i32 %272, %387
  %389 = add i32 %384, %388
  store i32 %389, ptr %351, align 4
  %390 = add nuw nsw i64 %240, %348
  %391 = getelementptr i32, ptr @buf4, i64 %390
  %.promoted15.1 = load i32, ptr %391, align 4
  %392 = mul i32 %275, %352
  %393 = add i32 %.promoted15.1, %392
  %394 = mul i32 %278, %357
  %395 = add i32 %393, %394
  %396 = mul i32 %281, %362
  %397 = add i32 %395, %396
  %398 = mul i32 %284, %367
  %399 = add i32 %397, %398
  %400 = mul i32 %287, %372
  %401 = add i32 %399, %400
  %402 = mul i32 %290, %377
  %403 = add i32 %401, %402
  %404 = mul i32 %293, %382
  %405 = add i32 %403, %404
  %406 = mul i32 %296, %387
  %407 = add i32 %405, %406
  store i32 %407, ptr %391, align 4
  %408 = add nuw nsw i64 %242, %348
  %409 = getelementptr i32, ptr @buf4, i64 %408
  %.promoted15.2 = load i32, ptr %409, align 4
  %410 = load i32, ptr %349, align 4
  %411 = mul i32 %299, %410
  %412 = add i32 %.promoted15.2, %411
  %413 = load i32, ptr %356, align 4
  %414 = mul i32 %302, %413
  %415 = add i32 %412, %414
  %416 = load i32, ptr %361, align 4
  %417 = mul i32 %305, %416
  %418 = add i32 %415, %417
  %419 = load i32, ptr %366, align 4
  %420 = mul i32 %308, %419
  %421 = add i32 %418, %420
  %422 = load i32, ptr %371, align 4
  %423 = mul i32 %311, %422
  %424 = add i32 %421, %423
  %425 = load i32, ptr %376, align 4
  %426 = mul i32 %314, %425
  %427 = add i32 %424, %426
  %428 = load i32, ptr %381, align 4
  %429 = mul i32 %317, %428
  %430 = add i32 %427, %429
  %431 = load i32, ptr %386, align 4
  %432 = mul i32 %320, %431
  %433 = add i32 %430, %432
  store i32 %433, ptr %409, align 4
  %434 = add nuw nsw i64 %244, %348
  %435 = getelementptr i32, ptr @buf4, i64 %434
  %.promoted15.3 = load i32, ptr %435, align 4
  %436 = mul i32 %323, %410
  %437 = add i32 %.promoted15.3, %436
  %438 = mul i32 %326, %413
  %439 = add i32 %437, %438
  %440 = mul i32 %329, %416
  %441 = add i32 %439, %440
  %442 = mul i32 %332, %419
  %443 = add i32 %441, %442
  %444 = mul i32 %335, %422
  %445 = add i32 %443, %444
  %446 = mul i32 %338, %425
  %447 = add i32 %445, %446
  %448 = mul i32 %341, %428
  %449 = add i32 %447, %448
  %450 = mul i32 %344, %431
  %451 = add i32 %449, %450
  store i32 %451, ptr %435, align 4
  %452 = add nuw nsw i64 %345, 1
  %453 = icmp ult i64 %345, 3
  br i1 %453, label %.preheader4, label %454

454:                                              ; preds = %.preheader4
  %455 = add nuw nsw i64 %245, 1
  %456 = icmp ult i64 %245, 3
  br i1 %456, label %.preheader6, label %457

457:                                              ; preds = %454
  %458 = add nuw nsw i64 %236, 1
  %459 = icmp ult i64 %236, 7
  br i1 %459, label %.preheader9, label %460

460:                                              ; preds = %457
  %461 = add nuw nsw i64 %233, 1
  %462 = icmp ult i64 %233, 7
  br i1 %462, label %.preheader12, label %463

463:                                              ; preds = %460
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: noreturn nounwind
define void @core_0_3() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %463, %0
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf9 to i64), i64 31), i64 0))
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4096) @buf9, i8 0, i64 4096, i1 false)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf8 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf7 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf9 to i64), i64 31), i64 0))
  br label %.preheader13

.preheader13:                                     ; preds = %1, %229
  %2 = phi i64 [ 0, %1 ], [ %230, %229 ]
  %3 = shl nuw nsw i64 %2, 5
  %4 = shl nuw nsw i64 %2, 4
  br label %.preheader10

.preheader10:                                     ; preds = %.preheader13, %226
  %5 = phi i64 [ 0, %.preheader13 ], [ %227, %226 ]
  %6 = shl nuw nsw i64 %5, 7
  %7 = add nuw nsw i64 %6, %4
  %8 = or i64 %6, 1
  %9 = add nuw nsw i64 %8, %4
  %10 = or i64 %6, 2
  %11 = add nuw nsw i64 %10, %4
  %12 = or i64 %6, 3
  %13 = add nuw nsw i64 %12, %4
  br label %.preheader7

.preheader7:                                      ; preds = %.preheader10, %223
  %14 = phi i64 [ 0, %.preheader10 ], [ %224, %223 ]
  %15 = shl nuw nsw i64 %14, 8
  %16 = add nuw nsw i64 %15, %3
  %17 = shl nuw nsw i64 %14, 5
  %18 = add nuw nsw i64 %6, %17
  %19 = getelementptr i32, ptr @buf7, i64 %18
  %20 = load i32, ptr %19, align 4
  %21 = or i64 %18, 4
  %22 = getelementptr i32, ptr @buf7, i64 %21
  %23 = load i32, ptr %22, align 4
  %24 = or i64 %18, 8
  %25 = getelementptr i32, ptr @buf7, i64 %24
  %26 = load i32, ptr %25, align 4
  %27 = or i64 %18, 12
  %28 = getelementptr i32, ptr @buf7, i64 %27
  %29 = load i32, ptr %28, align 4
  %30 = or i64 %18, 16
  %31 = getelementptr i32, ptr @buf7, i64 %30
  %32 = load i32, ptr %31, align 4
  %33 = or i64 %18, 20
  %34 = getelementptr i32, ptr @buf7, i64 %33
  %35 = load i32, ptr %34, align 4
  %36 = or i64 %18, 24
  %37 = getelementptr i32, ptr @buf7, i64 %36
  %38 = load i32, ptr %37, align 4
  %39 = or i64 %18, 28
  %40 = getelementptr i32, ptr @buf7, i64 %39
  %41 = load i32, ptr %40, align 4
  %42 = add nuw nsw i64 %8, %17
  %43 = getelementptr i32, ptr @buf7, i64 %42
  %44 = load i32, ptr %43, align 4
  %45 = or i64 %42, 4
  %46 = getelementptr i32, ptr @buf7, i64 %45
  %47 = load i32, ptr %46, align 4
  %48 = or i64 %42, 8
  %49 = getelementptr i32, ptr @buf7, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = or i64 %42, 12
  %52 = getelementptr i32, ptr @buf7, i64 %51
  %53 = load i32, ptr %52, align 4
  %54 = or i64 %42, 16
  %55 = getelementptr i32, ptr @buf7, i64 %54
  %56 = load i32, ptr %55, align 4
  %57 = or i64 %42, 20
  %58 = getelementptr i32, ptr @buf7, i64 %57
  %59 = load i32, ptr %58, align 4
  %60 = or i64 %42, 24
  %61 = getelementptr i32, ptr @buf7, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = or i64 %42, 28
  %64 = getelementptr i32, ptr @buf7, i64 %63
  %65 = load i32, ptr %64, align 4
  %66 = add nuw nsw i64 %10, %17
  %67 = getelementptr i32, ptr @buf7, i64 %66
  %68 = load i32, ptr %67, align 4
  %69 = or i64 %66, 4
  %70 = getelementptr i32, ptr @buf7, i64 %69
  %71 = load i32, ptr %70, align 4
  %72 = or i64 %66, 8
  %73 = getelementptr i32, ptr @buf7, i64 %72
  %74 = load i32, ptr %73, align 4
  %75 = or i64 %66, 12
  %76 = getelementptr i32, ptr @buf7, i64 %75
  %77 = load i32, ptr %76, align 4
  %78 = or i64 %66, 16
  %79 = getelementptr i32, ptr @buf7, i64 %78
  %80 = load i32, ptr %79, align 4
  %81 = or i64 %66, 20
  %82 = getelementptr i32, ptr @buf7, i64 %81
  %83 = load i32, ptr %82, align 4
  %84 = or i64 %66, 24
  %85 = getelementptr i32, ptr @buf7, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = or i64 %66, 28
  %88 = getelementptr i32, ptr @buf7, i64 %87
  %89 = load i32, ptr %88, align 4
  %90 = add nuw nsw i64 %12, %17
  %91 = getelementptr i32, ptr @buf7, i64 %90
  %92 = load i32, ptr %91, align 4
  %93 = or i64 %90, 4
  %94 = getelementptr i32, ptr @buf7, i64 %93
  %95 = load i32, ptr %94, align 4
  %96 = or i64 %90, 8
  %97 = getelementptr i32, ptr @buf7, i64 %96
  %98 = load i32, ptr %97, align 4
  %99 = or i64 %90, 12
  %100 = getelementptr i32, ptr @buf7, i64 %99
  %101 = load i32, ptr %100, align 4
  %102 = or i64 %90, 16
  %103 = getelementptr i32, ptr @buf7, i64 %102
  %104 = load i32, ptr %103, align 4
  %105 = or i64 %90, 20
  %106 = getelementptr i32, ptr @buf7, i64 %105
  %107 = load i32, ptr %106, align 4
  %108 = or i64 %90, 24
  %109 = getelementptr i32, ptr @buf7, i64 %108
  %110 = load i32, ptr %109, align 4
  %111 = or i64 %90, 28
  %112 = getelementptr i32, ptr @buf7, i64 %111
  %113 = load i32, ptr %112, align 4
  br label %.preheader5

.preheader5:                                      ; preds = %.preheader7, %.preheader5
  %114 = phi i64 [ 0, %.preheader7 ], [ %221, %.preheader5 ]
  %115 = shl nuw nsw i64 %114, 3
  %116 = add nuw nsw i64 %16, %115
  %117 = shl nuw nsw i64 %114, 2
  %118 = getelementptr i32, ptr @buf8, i64 %116
  %119 = add nuw nsw i64 %7, %117
  %120 = getelementptr i32, ptr @buf9, i64 %119
  %.promoted = load i32, ptr %120, align 4
  %121 = load i32, ptr %118, align 4
  %122 = mul i32 %20, %121
  %123 = add i32 %.promoted, %122
  %124 = or i64 %116, 1
  %125 = getelementptr i32, ptr @buf8, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = mul i32 %23, %126
  %128 = add i32 %123, %127
  %129 = or i64 %116, 2
  %130 = getelementptr i32, ptr @buf8, i64 %129
  %131 = load i32, ptr %130, align 4
  %132 = mul i32 %26, %131
  %133 = add i32 %128, %132
  %134 = or i64 %116, 3
  %135 = getelementptr i32, ptr @buf8, i64 %134
  %136 = load i32, ptr %135, align 4
  %137 = mul i32 %29, %136
  %138 = add i32 %133, %137
  %139 = or i64 %116, 4
  %140 = getelementptr i32, ptr @buf8, i64 %139
  %141 = load i32, ptr %140, align 4
  %142 = mul i32 %32, %141
  %143 = add i32 %138, %142
  %144 = or i64 %116, 5
  %145 = getelementptr i32, ptr @buf8, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %35, %146
  %148 = add i32 %143, %147
  %149 = or i64 %116, 6
  %150 = getelementptr i32, ptr @buf8, i64 %149
  %151 = load i32, ptr %150, align 4
  %152 = mul i32 %38, %151
  %153 = add i32 %148, %152
  %154 = or i64 %116, 7
  %155 = getelementptr i32, ptr @buf8, i64 %154
  %156 = load i32, ptr %155, align 4
  %157 = mul i32 %41, %156
  %158 = add i32 %153, %157
  store i32 %158, ptr %120, align 4
  %159 = add nuw nsw i64 %9, %117
  %160 = getelementptr i32, ptr @buf9, i64 %159
  %.promoted.1 = load i32, ptr %160, align 4
  %161 = mul i32 %44, %121
  %162 = add i32 %.promoted.1, %161
  %163 = mul i32 %47, %126
  %164 = add i32 %162, %163
  %165 = mul i32 %50, %131
  %166 = add i32 %164, %165
  %167 = mul i32 %53, %136
  %168 = add i32 %166, %167
  %169 = mul i32 %56, %141
  %170 = add i32 %168, %169
  %171 = mul i32 %59, %146
  %172 = add i32 %170, %171
  %173 = mul i32 %62, %151
  %174 = add i32 %172, %173
  %175 = mul i32 %65, %156
  %176 = add i32 %174, %175
  store i32 %176, ptr %160, align 4
  %177 = add nuw nsw i64 %11, %117
  %178 = getelementptr i32, ptr @buf9, i64 %177
  %.promoted.2 = load i32, ptr %178, align 4
  %179 = load i32, ptr %118, align 4
  %180 = mul i32 %68, %179
  %181 = add i32 %.promoted.2, %180
  %182 = load i32, ptr %125, align 4
  %183 = mul i32 %71, %182
  %184 = add i32 %181, %183
  %185 = load i32, ptr %130, align 4
  %186 = mul i32 %74, %185
  %187 = add i32 %184, %186
  %188 = load i32, ptr %135, align 4
  %189 = mul i32 %77, %188
  %190 = add i32 %187, %189
  %191 = load i32, ptr %140, align 4
  %192 = mul i32 %80, %191
  %193 = add i32 %190, %192
  %194 = load i32, ptr %145, align 4
  %195 = mul i32 %83, %194
  %196 = add i32 %193, %195
  %197 = load i32, ptr %150, align 4
  %198 = mul i32 %86, %197
  %199 = add i32 %196, %198
  %200 = load i32, ptr %155, align 4
  %201 = mul i32 %89, %200
  %202 = add i32 %199, %201
  store i32 %202, ptr %178, align 4
  %203 = add nuw nsw i64 %13, %117
  %204 = getelementptr i32, ptr @buf9, i64 %203
  %.promoted.3 = load i32, ptr %204, align 4
  %205 = mul i32 %92, %179
  %206 = add i32 %.promoted.3, %205
  %207 = mul i32 %95, %182
  %208 = add i32 %206, %207
  %209 = mul i32 %98, %185
  %210 = add i32 %208, %209
  %211 = mul i32 %101, %188
  %212 = add i32 %210, %211
  %213 = mul i32 %104, %191
  %214 = add i32 %212, %213
  %215 = mul i32 %107, %194
  %216 = add i32 %214, %215
  %217 = mul i32 %110, %197
  %218 = add i32 %216, %217
  %219 = mul i32 %113, %200
  %220 = add i32 %218, %219
  store i32 %220, ptr %204, align 4
  %221 = add nuw nsw i64 %114, 1
  %222 = icmp ult i64 %114, 3
  br i1 %222, label %.preheader5, label %223

223:                                              ; preds = %.preheader5
  %224 = add nuw nsw i64 %14, 1
  %225 = icmp ult i64 %14, 3
  br i1 %225, label %.preheader7, label %226

226:                                              ; preds = %223
  %227 = add nuw nsw i64 %5, 1
  %228 = icmp ult i64 %5, 7
  br i1 %228, label %.preheader10, label %229

229:                                              ; preds = %226
  %230 = add nuw nsw i64 %2, 1
  %231 = icmp ult i64 %2, 7
  br i1 %231, label %.preheader13, label %232

232:                                              ; preds = %229
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf6 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf5 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf9 to i64), i64 31), i64 0))
  br label %.preheader12

.preheader12:                                     ; preds = %232, %460
  %233 = phi i64 [ 0, %232 ], [ %461, %460 ]
  %234 = shl nuw nsw i64 %233, 5
  %235 = shl nuw nsw i64 %233, 4
  br label %.preheader9

.preheader9:                                      ; preds = %.preheader12, %457
  %236 = phi i64 [ 0, %.preheader12 ], [ %458, %457 ]
  %237 = shl nuw nsw i64 %236, 7
  %238 = add nuw nsw i64 %237, %235
  %239 = or i64 %237, 1
  %240 = add nuw nsw i64 %239, %235
  %241 = or i64 %237, 2
  %242 = add nuw nsw i64 %241, %235
  %243 = or i64 %237, 3
  %244 = add nuw nsw i64 %243, %235
  br label %.preheader6

.preheader6:                                      ; preds = %.preheader9, %454
  %245 = phi i64 [ 0, %.preheader9 ], [ %455, %454 ]
  %246 = shl nuw nsw i64 %245, 8
  %247 = add nuw nsw i64 %246, %234
  %248 = shl nuw nsw i64 %245, 5
  %249 = add nuw nsw i64 %237, %248
  %250 = getelementptr i32, ptr @buf5, i64 %249
  %251 = load i32, ptr %250, align 4
  %252 = or i64 %249, 4
  %253 = getelementptr i32, ptr @buf5, i64 %252
  %254 = load i32, ptr %253, align 4
  %255 = or i64 %249, 8
  %256 = getelementptr i32, ptr @buf5, i64 %255
  %257 = load i32, ptr %256, align 4
  %258 = or i64 %249, 12
  %259 = getelementptr i32, ptr @buf5, i64 %258
  %260 = load i32, ptr %259, align 4
  %261 = or i64 %249, 16
  %262 = getelementptr i32, ptr @buf5, i64 %261
  %263 = load i32, ptr %262, align 4
  %264 = or i64 %249, 20
  %265 = getelementptr i32, ptr @buf5, i64 %264
  %266 = load i32, ptr %265, align 4
  %267 = or i64 %249, 24
  %268 = getelementptr i32, ptr @buf5, i64 %267
  %269 = load i32, ptr %268, align 4
  %270 = or i64 %249, 28
  %271 = getelementptr i32, ptr @buf5, i64 %270
  %272 = load i32, ptr %271, align 4
  %273 = add nuw nsw i64 %239, %248
  %274 = getelementptr i32, ptr @buf5, i64 %273
  %275 = load i32, ptr %274, align 4
  %276 = or i64 %273, 4
  %277 = getelementptr i32, ptr @buf5, i64 %276
  %278 = load i32, ptr %277, align 4
  %279 = or i64 %273, 8
  %280 = getelementptr i32, ptr @buf5, i64 %279
  %281 = load i32, ptr %280, align 4
  %282 = or i64 %273, 12
  %283 = getelementptr i32, ptr @buf5, i64 %282
  %284 = load i32, ptr %283, align 4
  %285 = or i64 %273, 16
  %286 = getelementptr i32, ptr @buf5, i64 %285
  %287 = load i32, ptr %286, align 4
  %288 = or i64 %273, 20
  %289 = getelementptr i32, ptr @buf5, i64 %288
  %290 = load i32, ptr %289, align 4
  %291 = or i64 %273, 24
  %292 = getelementptr i32, ptr @buf5, i64 %291
  %293 = load i32, ptr %292, align 4
  %294 = or i64 %273, 28
  %295 = getelementptr i32, ptr @buf5, i64 %294
  %296 = load i32, ptr %295, align 4
  %297 = add nuw nsw i64 %241, %248
  %298 = getelementptr i32, ptr @buf5, i64 %297
  %299 = load i32, ptr %298, align 4
  %300 = or i64 %297, 4
  %301 = getelementptr i32, ptr @buf5, i64 %300
  %302 = load i32, ptr %301, align 4
  %303 = or i64 %297, 8
  %304 = getelementptr i32, ptr @buf5, i64 %303
  %305 = load i32, ptr %304, align 4
  %306 = or i64 %297, 12
  %307 = getelementptr i32, ptr @buf5, i64 %306
  %308 = load i32, ptr %307, align 4
  %309 = or i64 %297, 16
  %310 = getelementptr i32, ptr @buf5, i64 %309
  %311 = load i32, ptr %310, align 4
  %312 = or i64 %297, 20
  %313 = getelementptr i32, ptr @buf5, i64 %312
  %314 = load i32, ptr %313, align 4
  %315 = or i64 %297, 24
  %316 = getelementptr i32, ptr @buf5, i64 %315
  %317 = load i32, ptr %316, align 4
  %318 = or i64 %297, 28
  %319 = getelementptr i32, ptr @buf5, i64 %318
  %320 = load i32, ptr %319, align 4
  %321 = add nuw nsw i64 %243, %248
  %322 = getelementptr i32, ptr @buf5, i64 %321
  %323 = load i32, ptr %322, align 4
  %324 = or i64 %321, 4
  %325 = getelementptr i32, ptr @buf5, i64 %324
  %326 = load i32, ptr %325, align 4
  %327 = or i64 %321, 8
  %328 = getelementptr i32, ptr @buf5, i64 %327
  %329 = load i32, ptr %328, align 4
  %330 = or i64 %321, 12
  %331 = getelementptr i32, ptr @buf5, i64 %330
  %332 = load i32, ptr %331, align 4
  %333 = or i64 %321, 16
  %334 = getelementptr i32, ptr @buf5, i64 %333
  %335 = load i32, ptr %334, align 4
  %336 = or i64 %321, 20
  %337 = getelementptr i32, ptr @buf5, i64 %336
  %338 = load i32, ptr %337, align 4
  %339 = or i64 %321, 24
  %340 = getelementptr i32, ptr @buf5, i64 %339
  %341 = load i32, ptr %340, align 4
  %342 = or i64 %321, 28
  %343 = getelementptr i32, ptr @buf5, i64 %342
  %344 = load i32, ptr %343, align 4
  br label %.preheader4

.preheader4:                                      ; preds = %.preheader6, %.preheader4
  %345 = phi i64 [ 0, %.preheader6 ], [ %452, %.preheader4 ]
  %346 = shl nuw nsw i64 %345, 3
  %347 = add nuw nsw i64 %247, %346
  %348 = shl nuw nsw i64 %345, 2
  %349 = getelementptr i32, ptr @buf6, i64 %347
  %350 = add nuw nsw i64 %238, %348
  %351 = getelementptr i32, ptr @buf9, i64 %350
  %.promoted15 = load i32, ptr %351, align 4
  %352 = load i32, ptr %349, align 4
  %353 = mul i32 %251, %352
  %354 = add i32 %.promoted15, %353
  %355 = or i64 %347, 1
  %356 = getelementptr i32, ptr @buf6, i64 %355
  %357 = load i32, ptr %356, align 4
  %358 = mul i32 %254, %357
  %359 = add i32 %354, %358
  %360 = or i64 %347, 2
  %361 = getelementptr i32, ptr @buf6, i64 %360
  %362 = load i32, ptr %361, align 4
  %363 = mul i32 %257, %362
  %364 = add i32 %359, %363
  %365 = or i64 %347, 3
  %366 = getelementptr i32, ptr @buf6, i64 %365
  %367 = load i32, ptr %366, align 4
  %368 = mul i32 %260, %367
  %369 = add i32 %364, %368
  %370 = or i64 %347, 4
  %371 = getelementptr i32, ptr @buf6, i64 %370
  %372 = load i32, ptr %371, align 4
  %373 = mul i32 %263, %372
  %374 = add i32 %369, %373
  %375 = or i64 %347, 5
  %376 = getelementptr i32, ptr @buf6, i64 %375
  %377 = load i32, ptr %376, align 4
  %378 = mul i32 %266, %377
  %379 = add i32 %374, %378
  %380 = or i64 %347, 6
  %381 = getelementptr i32, ptr @buf6, i64 %380
  %382 = load i32, ptr %381, align 4
  %383 = mul i32 %269, %382
  %384 = add i32 %379, %383
  %385 = or i64 %347, 7
  %386 = getelementptr i32, ptr @buf6, i64 %385
  %387 = load i32, ptr %386, align 4
  %388 = mul i32 %272, %387
  %389 = add i32 %384, %388
  store i32 %389, ptr %351, align 4
  %390 = add nuw nsw i64 %240, %348
  %391 = getelementptr i32, ptr @buf9, i64 %390
  %.promoted15.1 = load i32, ptr %391, align 4
  %392 = mul i32 %275, %352
  %393 = add i32 %.promoted15.1, %392
  %394 = mul i32 %278, %357
  %395 = add i32 %393, %394
  %396 = mul i32 %281, %362
  %397 = add i32 %395, %396
  %398 = mul i32 %284, %367
  %399 = add i32 %397, %398
  %400 = mul i32 %287, %372
  %401 = add i32 %399, %400
  %402 = mul i32 %290, %377
  %403 = add i32 %401, %402
  %404 = mul i32 %293, %382
  %405 = add i32 %403, %404
  %406 = mul i32 %296, %387
  %407 = add i32 %405, %406
  store i32 %407, ptr %391, align 4
  %408 = add nuw nsw i64 %242, %348
  %409 = getelementptr i32, ptr @buf9, i64 %408
  %.promoted15.2 = load i32, ptr %409, align 4
  %410 = load i32, ptr %349, align 4
  %411 = mul i32 %299, %410
  %412 = add i32 %.promoted15.2, %411
  %413 = load i32, ptr %356, align 4
  %414 = mul i32 %302, %413
  %415 = add i32 %412, %414
  %416 = load i32, ptr %361, align 4
  %417 = mul i32 %305, %416
  %418 = add i32 %415, %417
  %419 = load i32, ptr %366, align 4
  %420 = mul i32 %308, %419
  %421 = add i32 %418, %420
  %422 = load i32, ptr %371, align 4
  %423 = mul i32 %311, %422
  %424 = add i32 %421, %423
  %425 = load i32, ptr %376, align 4
  %426 = mul i32 %314, %425
  %427 = add i32 %424, %426
  %428 = load i32, ptr %381, align 4
  %429 = mul i32 %317, %428
  %430 = add i32 %427, %429
  %431 = load i32, ptr %386, align 4
  %432 = mul i32 %320, %431
  %433 = add i32 %430, %432
  store i32 %433, ptr %409, align 4
  %434 = add nuw nsw i64 %244, %348
  %435 = getelementptr i32, ptr @buf9, i64 %434
  %.promoted15.3 = load i32, ptr %435, align 4
  %436 = mul i32 %323, %410
  %437 = add i32 %.promoted15.3, %436
  %438 = mul i32 %326, %413
  %439 = add i32 %437, %438
  %440 = mul i32 %329, %416
  %441 = add i32 %439, %440
  %442 = mul i32 %332, %419
  %443 = add i32 %441, %442
  %444 = mul i32 %335, %422
  %445 = add i32 %443, %444
  %446 = mul i32 %338, %425
  %447 = add i32 %445, %446
  %448 = mul i32 %341, %428
  %449 = add i32 %447, %448
  %450 = mul i32 %344, %431
  %451 = add i32 %449, %450
  store i32 %451, ptr %435, align 4
  %452 = add nuw nsw i64 %345, 1
  %453 = icmp ult i64 %345, 3
  br i1 %453, label %.preheader4, label %454

454:                                              ; preds = %.preheader4
  %455 = add nuw nsw i64 %245, 1
  %456 = icmp ult i64 %245, 3
  br i1 %456, label %.preheader6, label %457

457:                                              ; preds = %454
  %458 = add nuw nsw i64 %236, 1
  %459 = icmp ult i64 %236, 7
  br i1 %459, label %.preheader9, label %460

460:                                              ; preds = %457
  %461 = add nuw nsw i64 %233, 1
  %462 = icmp ult i64 %233, 7
  br i1 %462, label %.preheader12, label %463

463:                                              ; preds = %460
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: noreturn nounwind
define void @core_0_4() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %463, %0
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf14 to i64), i64 31), i64 0))
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4096) @buf14, i8 0, i64 4096, i1 false)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf13 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf12 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf14 to i64), i64 31), i64 0))
  br label %.preheader13

.preheader13:                                     ; preds = %1, %229
  %2 = phi i64 [ 0, %1 ], [ %230, %229 ]
  %3 = shl nuw nsw i64 %2, 5
  %4 = shl nuw nsw i64 %2, 4
  br label %.preheader10

.preheader10:                                     ; preds = %.preheader13, %226
  %5 = phi i64 [ 0, %.preheader13 ], [ %227, %226 ]
  %6 = shl nuw nsw i64 %5, 7
  %7 = add nuw nsw i64 %6, %4
  %8 = or i64 %6, 1
  %9 = add nuw nsw i64 %8, %4
  %10 = or i64 %6, 2
  %11 = add nuw nsw i64 %10, %4
  %12 = or i64 %6, 3
  %13 = add nuw nsw i64 %12, %4
  br label %.preheader7

.preheader7:                                      ; preds = %.preheader10, %223
  %14 = phi i64 [ 0, %.preheader10 ], [ %224, %223 ]
  %15 = shl nuw nsw i64 %14, 8
  %16 = add nuw nsw i64 %15, %3
  %17 = shl nuw nsw i64 %14, 5
  %18 = add nuw nsw i64 %6, %17
  %19 = getelementptr i32, ptr @buf12, i64 %18
  %20 = load i32, ptr %19, align 4
  %21 = or i64 %18, 4
  %22 = getelementptr i32, ptr @buf12, i64 %21
  %23 = load i32, ptr %22, align 4
  %24 = or i64 %18, 8
  %25 = getelementptr i32, ptr @buf12, i64 %24
  %26 = load i32, ptr %25, align 4
  %27 = or i64 %18, 12
  %28 = getelementptr i32, ptr @buf12, i64 %27
  %29 = load i32, ptr %28, align 4
  %30 = or i64 %18, 16
  %31 = getelementptr i32, ptr @buf12, i64 %30
  %32 = load i32, ptr %31, align 4
  %33 = or i64 %18, 20
  %34 = getelementptr i32, ptr @buf12, i64 %33
  %35 = load i32, ptr %34, align 4
  %36 = or i64 %18, 24
  %37 = getelementptr i32, ptr @buf12, i64 %36
  %38 = load i32, ptr %37, align 4
  %39 = or i64 %18, 28
  %40 = getelementptr i32, ptr @buf12, i64 %39
  %41 = load i32, ptr %40, align 4
  %42 = add nuw nsw i64 %8, %17
  %43 = getelementptr i32, ptr @buf12, i64 %42
  %44 = load i32, ptr %43, align 4
  %45 = or i64 %42, 4
  %46 = getelementptr i32, ptr @buf12, i64 %45
  %47 = load i32, ptr %46, align 4
  %48 = or i64 %42, 8
  %49 = getelementptr i32, ptr @buf12, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = or i64 %42, 12
  %52 = getelementptr i32, ptr @buf12, i64 %51
  %53 = load i32, ptr %52, align 4
  %54 = or i64 %42, 16
  %55 = getelementptr i32, ptr @buf12, i64 %54
  %56 = load i32, ptr %55, align 4
  %57 = or i64 %42, 20
  %58 = getelementptr i32, ptr @buf12, i64 %57
  %59 = load i32, ptr %58, align 4
  %60 = or i64 %42, 24
  %61 = getelementptr i32, ptr @buf12, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = or i64 %42, 28
  %64 = getelementptr i32, ptr @buf12, i64 %63
  %65 = load i32, ptr %64, align 4
  %66 = add nuw nsw i64 %10, %17
  %67 = getelementptr i32, ptr @buf12, i64 %66
  %68 = load i32, ptr %67, align 4
  %69 = or i64 %66, 4
  %70 = getelementptr i32, ptr @buf12, i64 %69
  %71 = load i32, ptr %70, align 4
  %72 = or i64 %66, 8
  %73 = getelementptr i32, ptr @buf12, i64 %72
  %74 = load i32, ptr %73, align 4
  %75 = or i64 %66, 12
  %76 = getelementptr i32, ptr @buf12, i64 %75
  %77 = load i32, ptr %76, align 4
  %78 = or i64 %66, 16
  %79 = getelementptr i32, ptr @buf12, i64 %78
  %80 = load i32, ptr %79, align 4
  %81 = or i64 %66, 20
  %82 = getelementptr i32, ptr @buf12, i64 %81
  %83 = load i32, ptr %82, align 4
  %84 = or i64 %66, 24
  %85 = getelementptr i32, ptr @buf12, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = or i64 %66, 28
  %88 = getelementptr i32, ptr @buf12, i64 %87
  %89 = load i32, ptr %88, align 4
  %90 = add nuw nsw i64 %12, %17
  %91 = getelementptr i32, ptr @buf12, i64 %90
  %92 = load i32, ptr %91, align 4
  %93 = or i64 %90, 4
  %94 = getelementptr i32, ptr @buf12, i64 %93
  %95 = load i32, ptr %94, align 4
  %96 = or i64 %90, 8
  %97 = getelementptr i32, ptr @buf12, i64 %96
  %98 = load i32, ptr %97, align 4
  %99 = or i64 %90, 12
  %100 = getelementptr i32, ptr @buf12, i64 %99
  %101 = load i32, ptr %100, align 4
  %102 = or i64 %90, 16
  %103 = getelementptr i32, ptr @buf12, i64 %102
  %104 = load i32, ptr %103, align 4
  %105 = or i64 %90, 20
  %106 = getelementptr i32, ptr @buf12, i64 %105
  %107 = load i32, ptr %106, align 4
  %108 = or i64 %90, 24
  %109 = getelementptr i32, ptr @buf12, i64 %108
  %110 = load i32, ptr %109, align 4
  %111 = or i64 %90, 28
  %112 = getelementptr i32, ptr @buf12, i64 %111
  %113 = load i32, ptr %112, align 4
  br label %.preheader5

.preheader5:                                      ; preds = %.preheader7, %.preheader5
  %114 = phi i64 [ 0, %.preheader7 ], [ %221, %.preheader5 ]
  %115 = shl nuw nsw i64 %114, 3
  %116 = add nuw nsw i64 %16, %115
  %117 = shl nuw nsw i64 %114, 2
  %118 = getelementptr i32, ptr @buf13, i64 %116
  %119 = add nuw nsw i64 %7, %117
  %120 = getelementptr i32, ptr @buf14, i64 %119
  %.promoted = load i32, ptr %120, align 4
  %121 = load i32, ptr %118, align 4
  %122 = mul i32 %20, %121
  %123 = add i32 %.promoted, %122
  %124 = or i64 %116, 1
  %125 = getelementptr i32, ptr @buf13, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = mul i32 %23, %126
  %128 = add i32 %123, %127
  %129 = or i64 %116, 2
  %130 = getelementptr i32, ptr @buf13, i64 %129
  %131 = load i32, ptr %130, align 4
  %132 = mul i32 %26, %131
  %133 = add i32 %128, %132
  %134 = or i64 %116, 3
  %135 = getelementptr i32, ptr @buf13, i64 %134
  %136 = load i32, ptr %135, align 4
  %137 = mul i32 %29, %136
  %138 = add i32 %133, %137
  %139 = or i64 %116, 4
  %140 = getelementptr i32, ptr @buf13, i64 %139
  %141 = load i32, ptr %140, align 4
  %142 = mul i32 %32, %141
  %143 = add i32 %138, %142
  %144 = or i64 %116, 5
  %145 = getelementptr i32, ptr @buf13, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %35, %146
  %148 = add i32 %143, %147
  %149 = or i64 %116, 6
  %150 = getelementptr i32, ptr @buf13, i64 %149
  %151 = load i32, ptr %150, align 4
  %152 = mul i32 %38, %151
  %153 = add i32 %148, %152
  %154 = or i64 %116, 7
  %155 = getelementptr i32, ptr @buf13, i64 %154
  %156 = load i32, ptr %155, align 4
  %157 = mul i32 %41, %156
  %158 = add i32 %153, %157
  store i32 %158, ptr %120, align 4
  %159 = add nuw nsw i64 %9, %117
  %160 = getelementptr i32, ptr @buf14, i64 %159
  %.promoted.1 = load i32, ptr %160, align 4
  %161 = mul i32 %44, %121
  %162 = add i32 %.promoted.1, %161
  %163 = mul i32 %47, %126
  %164 = add i32 %162, %163
  %165 = mul i32 %50, %131
  %166 = add i32 %164, %165
  %167 = mul i32 %53, %136
  %168 = add i32 %166, %167
  %169 = mul i32 %56, %141
  %170 = add i32 %168, %169
  %171 = mul i32 %59, %146
  %172 = add i32 %170, %171
  %173 = mul i32 %62, %151
  %174 = add i32 %172, %173
  %175 = mul i32 %65, %156
  %176 = add i32 %174, %175
  store i32 %176, ptr %160, align 4
  %177 = add nuw nsw i64 %11, %117
  %178 = getelementptr i32, ptr @buf14, i64 %177
  %.promoted.2 = load i32, ptr %178, align 4
  %179 = load i32, ptr %118, align 4
  %180 = mul i32 %68, %179
  %181 = add i32 %.promoted.2, %180
  %182 = load i32, ptr %125, align 4
  %183 = mul i32 %71, %182
  %184 = add i32 %181, %183
  %185 = load i32, ptr %130, align 4
  %186 = mul i32 %74, %185
  %187 = add i32 %184, %186
  %188 = load i32, ptr %135, align 4
  %189 = mul i32 %77, %188
  %190 = add i32 %187, %189
  %191 = load i32, ptr %140, align 4
  %192 = mul i32 %80, %191
  %193 = add i32 %190, %192
  %194 = load i32, ptr %145, align 4
  %195 = mul i32 %83, %194
  %196 = add i32 %193, %195
  %197 = load i32, ptr %150, align 4
  %198 = mul i32 %86, %197
  %199 = add i32 %196, %198
  %200 = load i32, ptr %155, align 4
  %201 = mul i32 %89, %200
  %202 = add i32 %199, %201
  store i32 %202, ptr %178, align 4
  %203 = add nuw nsw i64 %13, %117
  %204 = getelementptr i32, ptr @buf14, i64 %203
  %.promoted.3 = load i32, ptr %204, align 4
  %205 = mul i32 %92, %179
  %206 = add i32 %.promoted.3, %205
  %207 = mul i32 %95, %182
  %208 = add i32 %206, %207
  %209 = mul i32 %98, %185
  %210 = add i32 %208, %209
  %211 = mul i32 %101, %188
  %212 = add i32 %210, %211
  %213 = mul i32 %104, %191
  %214 = add i32 %212, %213
  %215 = mul i32 %107, %194
  %216 = add i32 %214, %215
  %217 = mul i32 %110, %197
  %218 = add i32 %216, %217
  %219 = mul i32 %113, %200
  %220 = add i32 %218, %219
  store i32 %220, ptr %204, align 4
  %221 = add nuw nsw i64 %114, 1
  %222 = icmp ult i64 %114, 3
  br i1 %222, label %.preheader5, label %223

223:                                              ; preds = %.preheader5
  %224 = add nuw nsw i64 %14, 1
  %225 = icmp ult i64 %14, 3
  br i1 %225, label %.preheader7, label %226

226:                                              ; preds = %223
  %227 = add nuw nsw i64 %5, 1
  %228 = icmp ult i64 %5, 7
  br i1 %228, label %.preheader10, label %229

229:                                              ; preds = %226
  %230 = add nuw nsw i64 %2, 1
  %231 = icmp ult i64 %2, 7
  br i1 %231, label %.preheader13, label %232

232:                                              ; preds = %229
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf11 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf10 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf14 to i64), i64 31), i64 0))
  br label %.preheader12

.preheader12:                                     ; preds = %232, %460
  %233 = phi i64 [ 0, %232 ], [ %461, %460 ]
  %234 = shl nuw nsw i64 %233, 5
  %235 = shl nuw nsw i64 %233, 4
  br label %.preheader9

.preheader9:                                      ; preds = %.preheader12, %457
  %236 = phi i64 [ 0, %.preheader12 ], [ %458, %457 ]
  %237 = shl nuw nsw i64 %236, 7
  %238 = add nuw nsw i64 %237, %235
  %239 = or i64 %237, 1
  %240 = add nuw nsw i64 %239, %235
  %241 = or i64 %237, 2
  %242 = add nuw nsw i64 %241, %235
  %243 = or i64 %237, 3
  %244 = add nuw nsw i64 %243, %235
  br label %.preheader6

.preheader6:                                      ; preds = %.preheader9, %454
  %245 = phi i64 [ 0, %.preheader9 ], [ %455, %454 ]
  %246 = shl nuw nsw i64 %245, 8
  %247 = add nuw nsw i64 %246, %234
  %248 = shl nuw nsw i64 %245, 5
  %249 = add nuw nsw i64 %237, %248
  %250 = getelementptr i32, ptr @buf10, i64 %249
  %251 = load i32, ptr %250, align 4
  %252 = or i64 %249, 4
  %253 = getelementptr i32, ptr @buf10, i64 %252
  %254 = load i32, ptr %253, align 4
  %255 = or i64 %249, 8
  %256 = getelementptr i32, ptr @buf10, i64 %255
  %257 = load i32, ptr %256, align 4
  %258 = or i64 %249, 12
  %259 = getelementptr i32, ptr @buf10, i64 %258
  %260 = load i32, ptr %259, align 4
  %261 = or i64 %249, 16
  %262 = getelementptr i32, ptr @buf10, i64 %261
  %263 = load i32, ptr %262, align 4
  %264 = or i64 %249, 20
  %265 = getelementptr i32, ptr @buf10, i64 %264
  %266 = load i32, ptr %265, align 4
  %267 = or i64 %249, 24
  %268 = getelementptr i32, ptr @buf10, i64 %267
  %269 = load i32, ptr %268, align 4
  %270 = or i64 %249, 28
  %271 = getelementptr i32, ptr @buf10, i64 %270
  %272 = load i32, ptr %271, align 4
  %273 = add nuw nsw i64 %239, %248
  %274 = getelementptr i32, ptr @buf10, i64 %273
  %275 = load i32, ptr %274, align 4
  %276 = or i64 %273, 4
  %277 = getelementptr i32, ptr @buf10, i64 %276
  %278 = load i32, ptr %277, align 4
  %279 = or i64 %273, 8
  %280 = getelementptr i32, ptr @buf10, i64 %279
  %281 = load i32, ptr %280, align 4
  %282 = or i64 %273, 12
  %283 = getelementptr i32, ptr @buf10, i64 %282
  %284 = load i32, ptr %283, align 4
  %285 = or i64 %273, 16
  %286 = getelementptr i32, ptr @buf10, i64 %285
  %287 = load i32, ptr %286, align 4
  %288 = or i64 %273, 20
  %289 = getelementptr i32, ptr @buf10, i64 %288
  %290 = load i32, ptr %289, align 4
  %291 = or i64 %273, 24
  %292 = getelementptr i32, ptr @buf10, i64 %291
  %293 = load i32, ptr %292, align 4
  %294 = or i64 %273, 28
  %295 = getelementptr i32, ptr @buf10, i64 %294
  %296 = load i32, ptr %295, align 4
  %297 = add nuw nsw i64 %241, %248
  %298 = getelementptr i32, ptr @buf10, i64 %297
  %299 = load i32, ptr %298, align 4
  %300 = or i64 %297, 4
  %301 = getelementptr i32, ptr @buf10, i64 %300
  %302 = load i32, ptr %301, align 4
  %303 = or i64 %297, 8
  %304 = getelementptr i32, ptr @buf10, i64 %303
  %305 = load i32, ptr %304, align 4
  %306 = or i64 %297, 12
  %307 = getelementptr i32, ptr @buf10, i64 %306
  %308 = load i32, ptr %307, align 4
  %309 = or i64 %297, 16
  %310 = getelementptr i32, ptr @buf10, i64 %309
  %311 = load i32, ptr %310, align 4
  %312 = or i64 %297, 20
  %313 = getelementptr i32, ptr @buf10, i64 %312
  %314 = load i32, ptr %313, align 4
  %315 = or i64 %297, 24
  %316 = getelementptr i32, ptr @buf10, i64 %315
  %317 = load i32, ptr %316, align 4
  %318 = or i64 %297, 28
  %319 = getelementptr i32, ptr @buf10, i64 %318
  %320 = load i32, ptr %319, align 4
  %321 = add nuw nsw i64 %243, %248
  %322 = getelementptr i32, ptr @buf10, i64 %321
  %323 = load i32, ptr %322, align 4
  %324 = or i64 %321, 4
  %325 = getelementptr i32, ptr @buf10, i64 %324
  %326 = load i32, ptr %325, align 4
  %327 = or i64 %321, 8
  %328 = getelementptr i32, ptr @buf10, i64 %327
  %329 = load i32, ptr %328, align 4
  %330 = or i64 %321, 12
  %331 = getelementptr i32, ptr @buf10, i64 %330
  %332 = load i32, ptr %331, align 4
  %333 = or i64 %321, 16
  %334 = getelementptr i32, ptr @buf10, i64 %333
  %335 = load i32, ptr %334, align 4
  %336 = or i64 %321, 20
  %337 = getelementptr i32, ptr @buf10, i64 %336
  %338 = load i32, ptr %337, align 4
  %339 = or i64 %321, 24
  %340 = getelementptr i32, ptr @buf10, i64 %339
  %341 = load i32, ptr %340, align 4
  %342 = or i64 %321, 28
  %343 = getelementptr i32, ptr @buf10, i64 %342
  %344 = load i32, ptr %343, align 4
  br label %.preheader4

.preheader4:                                      ; preds = %.preheader6, %.preheader4
  %345 = phi i64 [ 0, %.preheader6 ], [ %452, %.preheader4 ]
  %346 = shl nuw nsw i64 %345, 3
  %347 = add nuw nsw i64 %247, %346
  %348 = shl nuw nsw i64 %345, 2
  %349 = getelementptr i32, ptr @buf11, i64 %347
  %350 = add nuw nsw i64 %238, %348
  %351 = getelementptr i32, ptr @buf14, i64 %350
  %.promoted15 = load i32, ptr %351, align 4
  %352 = load i32, ptr %349, align 4
  %353 = mul i32 %251, %352
  %354 = add i32 %.promoted15, %353
  %355 = or i64 %347, 1
  %356 = getelementptr i32, ptr @buf11, i64 %355
  %357 = load i32, ptr %356, align 4
  %358 = mul i32 %254, %357
  %359 = add i32 %354, %358
  %360 = or i64 %347, 2
  %361 = getelementptr i32, ptr @buf11, i64 %360
  %362 = load i32, ptr %361, align 4
  %363 = mul i32 %257, %362
  %364 = add i32 %359, %363
  %365 = or i64 %347, 3
  %366 = getelementptr i32, ptr @buf11, i64 %365
  %367 = load i32, ptr %366, align 4
  %368 = mul i32 %260, %367
  %369 = add i32 %364, %368
  %370 = or i64 %347, 4
  %371 = getelementptr i32, ptr @buf11, i64 %370
  %372 = load i32, ptr %371, align 4
  %373 = mul i32 %263, %372
  %374 = add i32 %369, %373
  %375 = or i64 %347, 5
  %376 = getelementptr i32, ptr @buf11, i64 %375
  %377 = load i32, ptr %376, align 4
  %378 = mul i32 %266, %377
  %379 = add i32 %374, %378
  %380 = or i64 %347, 6
  %381 = getelementptr i32, ptr @buf11, i64 %380
  %382 = load i32, ptr %381, align 4
  %383 = mul i32 %269, %382
  %384 = add i32 %379, %383
  %385 = or i64 %347, 7
  %386 = getelementptr i32, ptr @buf11, i64 %385
  %387 = load i32, ptr %386, align 4
  %388 = mul i32 %272, %387
  %389 = add i32 %384, %388
  store i32 %389, ptr %351, align 4
  %390 = add nuw nsw i64 %240, %348
  %391 = getelementptr i32, ptr @buf14, i64 %390
  %.promoted15.1 = load i32, ptr %391, align 4
  %392 = mul i32 %275, %352
  %393 = add i32 %.promoted15.1, %392
  %394 = mul i32 %278, %357
  %395 = add i32 %393, %394
  %396 = mul i32 %281, %362
  %397 = add i32 %395, %396
  %398 = mul i32 %284, %367
  %399 = add i32 %397, %398
  %400 = mul i32 %287, %372
  %401 = add i32 %399, %400
  %402 = mul i32 %290, %377
  %403 = add i32 %401, %402
  %404 = mul i32 %293, %382
  %405 = add i32 %403, %404
  %406 = mul i32 %296, %387
  %407 = add i32 %405, %406
  store i32 %407, ptr %391, align 4
  %408 = add nuw nsw i64 %242, %348
  %409 = getelementptr i32, ptr @buf14, i64 %408
  %.promoted15.2 = load i32, ptr %409, align 4
  %410 = load i32, ptr %349, align 4
  %411 = mul i32 %299, %410
  %412 = add i32 %.promoted15.2, %411
  %413 = load i32, ptr %356, align 4
  %414 = mul i32 %302, %413
  %415 = add i32 %412, %414
  %416 = load i32, ptr %361, align 4
  %417 = mul i32 %305, %416
  %418 = add i32 %415, %417
  %419 = load i32, ptr %366, align 4
  %420 = mul i32 %308, %419
  %421 = add i32 %418, %420
  %422 = load i32, ptr %371, align 4
  %423 = mul i32 %311, %422
  %424 = add i32 %421, %423
  %425 = load i32, ptr %376, align 4
  %426 = mul i32 %314, %425
  %427 = add i32 %424, %426
  %428 = load i32, ptr %381, align 4
  %429 = mul i32 %317, %428
  %430 = add i32 %427, %429
  %431 = load i32, ptr %386, align 4
  %432 = mul i32 %320, %431
  %433 = add i32 %430, %432
  store i32 %433, ptr %409, align 4
  %434 = add nuw nsw i64 %244, %348
  %435 = getelementptr i32, ptr @buf14, i64 %434
  %.promoted15.3 = load i32, ptr %435, align 4
  %436 = mul i32 %323, %410
  %437 = add i32 %.promoted15.3, %436
  %438 = mul i32 %326, %413
  %439 = add i32 %437, %438
  %440 = mul i32 %329, %416
  %441 = add i32 %439, %440
  %442 = mul i32 %332, %419
  %443 = add i32 %441, %442
  %444 = mul i32 %335, %422
  %445 = add i32 %443, %444
  %446 = mul i32 %338, %425
  %447 = add i32 %445, %446
  %448 = mul i32 %341, %428
  %449 = add i32 %447, %448
  %450 = mul i32 %344, %431
  %451 = add i32 %449, %450
  store i32 %451, ptr %435, align 4
  %452 = add nuw nsw i64 %345, 1
  %453 = icmp ult i64 %345, 3
  br i1 %453, label %.preheader4, label %454

454:                                              ; preds = %.preheader4
  %455 = add nuw nsw i64 %245, 1
  %456 = icmp ult i64 %245, 3
  br i1 %456, label %.preheader6, label %457

457:                                              ; preds = %454
  %458 = add nuw nsw i64 %236, 1
  %459 = icmp ult i64 %236, 7
  br i1 %459, label %.preheader9, label %460

460:                                              ; preds = %457
  %461 = add nuw nsw i64 %233, 1
  %462 = icmp ult i64 %233, 7
  br i1 %462, label %.preheader12, label %463

463:                                              ; preds = %460
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: noreturn nounwind
define void @core_0_5() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %463, %0
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf19 to i64), i64 31), i64 0))
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4096) @buf19, i8 0, i64 4096, i1 false)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf18 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf17 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf19 to i64), i64 31), i64 0))
  br label %.preheader13

.preheader13:                                     ; preds = %1, %229
  %2 = phi i64 [ 0, %1 ], [ %230, %229 ]
  %3 = shl nuw nsw i64 %2, 5
  %4 = shl nuw nsw i64 %2, 4
  br label %.preheader10

.preheader10:                                     ; preds = %.preheader13, %226
  %5 = phi i64 [ 0, %.preheader13 ], [ %227, %226 ]
  %6 = shl nuw nsw i64 %5, 7
  %7 = add nuw nsw i64 %6, %4
  %8 = or i64 %6, 1
  %9 = add nuw nsw i64 %8, %4
  %10 = or i64 %6, 2
  %11 = add nuw nsw i64 %10, %4
  %12 = or i64 %6, 3
  %13 = add nuw nsw i64 %12, %4
  br label %.preheader7

.preheader7:                                      ; preds = %.preheader10, %223
  %14 = phi i64 [ 0, %.preheader10 ], [ %224, %223 ]
  %15 = shl nuw nsw i64 %14, 8
  %16 = add nuw nsw i64 %15, %3
  %17 = shl nuw nsw i64 %14, 5
  %18 = add nuw nsw i64 %6, %17
  %19 = getelementptr i32, ptr @buf17, i64 %18
  %20 = load i32, ptr %19, align 4
  %21 = or i64 %18, 4
  %22 = getelementptr i32, ptr @buf17, i64 %21
  %23 = load i32, ptr %22, align 4
  %24 = or i64 %18, 8
  %25 = getelementptr i32, ptr @buf17, i64 %24
  %26 = load i32, ptr %25, align 4
  %27 = or i64 %18, 12
  %28 = getelementptr i32, ptr @buf17, i64 %27
  %29 = load i32, ptr %28, align 4
  %30 = or i64 %18, 16
  %31 = getelementptr i32, ptr @buf17, i64 %30
  %32 = load i32, ptr %31, align 4
  %33 = or i64 %18, 20
  %34 = getelementptr i32, ptr @buf17, i64 %33
  %35 = load i32, ptr %34, align 4
  %36 = or i64 %18, 24
  %37 = getelementptr i32, ptr @buf17, i64 %36
  %38 = load i32, ptr %37, align 4
  %39 = or i64 %18, 28
  %40 = getelementptr i32, ptr @buf17, i64 %39
  %41 = load i32, ptr %40, align 4
  %42 = add nuw nsw i64 %8, %17
  %43 = getelementptr i32, ptr @buf17, i64 %42
  %44 = load i32, ptr %43, align 4
  %45 = or i64 %42, 4
  %46 = getelementptr i32, ptr @buf17, i64 %45
  %47 = load i32, ptr %46, align 4
  %48 = or i64 %42, 8
  %49 = getelementptr i32, ptr @buf17, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = or i64 %42, 12
  %52 = getelementptr i32, ptr @buf17, i64 %51
  %53 = load i32, ptr %52, align 4
  %54 = or i64 %42, 16
  %55 = getelementptr i32, ptr @buf17, i64 %54
  %56 = load i32, ptr %55, align 4
  %57 = or i64 %42, 20
  %58 = getelementptr i32, ptr @buf17, i64 %57
  %59 = load i32, ptr %58, align 4
  %60 = or i64 %42, 24
  %61 = getelementptr i32, ptr @buf17, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = or i64 %42, 28
  %64 = getelementptr i32, ptr @buf17, i64 %63
  %65 = load i32, ptr %64, align 4
  %66 = add nuw nsw i64 %10, %17
  %67 = getelementptr i32, ptr @buf17, i64 %66
  %68 = load i32, ptr %67, align 4
  %69 = or i64 %66, 4
  %70 = getelementptr i32, ptr @buf17, i64 %69
  %71 = load i32, ptr %70, align 4
  %72 = or i64 %66, 8
  %73 = getelementptr i32, ptr @buf17, i64 %72
  %74 = load i32, ptr %73, align 4
  %75 = or i64 %66, 12
  %76 = getelementptr i32, ptr @buf17, i64 %75
  %77 = load i32, ptr %76, align 4
  %78 = or i64 %66, 16
  %79 = getelementptr i32, ptr @buf17, i64 %78
  %80 = load i32, ptr %79, align 4
  %81 = or i64 %66, 20
  %82 = getelementptr i32, ptr @buf17, i64 %81
  %83 = load i32, ptr %82, align 4
  %84 = or i64 %66, 24
  %85 = getelementptr i32, ptr @buf17, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = or i64 %66, 28
  %88 = getelementptr i32, ptr @buf17, i64 %87
  %89 = load i32, ptr %88, align 4
  %90 = add nuw nsw i64 %12, %17
  %91 = getelementptr i32, ptr @buf17, i64 %90
  %92 = load i32, ptr %91, align 4
  %93 = or i64 %90, 4
  %94 = getelementptr i32, ptr @buf17, i64 %93
  %95 = load i32, ptr %94, align 4
  %96 = or i64 %90, 8
  %97 = getelementptr i32, ptr @buf17, i64 %96
  %98 = load i32, ptr %97, align 4
  %99 = or i64 %90, 12
  %100 = getelementptr i32, ptr @buf17, i64 %99
  %101 = load i32, ptr %100, align 4
  %102 = or i64 %90, 16
  %103 = getelementptr i32, ptr @buf17, i64 %102
  %104 = load i32, ptr %103, align 4
  %105 = or i64 %90, 20
  %106 = getelementptr i32, ptr @buf17, i64 %105
  %107 = load i32, ptr %106, align 4
  %108 = or i64 %90, 24
  %109 = getelementptr i32, ptr @buf17, i64 %108
  %110 = load i32, ptr %109, align 4
  %111 = or i64 %90, 28
  %112 = getelementptr i32, ptr @buf17, i64 %111
  %113 = load i32, ptr %112, align 4
  br label %.preheader5

.preheader5:                                      ; preds = %.preheader7, %.preheader5
  %114 = phi i64 [ 0, %.preheader7 ], [ %221, %.preheader5 ]
  %115 = shl nuw nsw i64 %114, 3
  %116 = add nuw nsw i64 %16, %115
  %117 = shl nuw nsw i64 %114, 2
  %118 = getelementptr i32, ptr @buf18, i64 %116
  %119 = add nuw nsw i64 %7, %117
  %120 = getelementptr i32, ptr @buf19, i64 %119
  %.promoted = load i32, ptr %120, align 4
  %121 = load i32, ptr %118, align 4
  %122 = mul i32 %20, %121
  %123 = add i32 %.promoted, %122
  %124 = or i64 %116, 1
  %125 = getelementptr i32, ptr @buf18, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = mul i32 %23, %126
  %128 = add i32 %123, %127
  %129 = or i64 %116, 2
  %130 = getelementptr i32, ptr @buf18, i64 %129
  %131 = load i32, ptr %130, align 4
  %132 = mul i32 %26, %131
  %133 = add i32 %128, %132
  %134 = or i64 %116, 3
  %135 = getelementptr i32, ptr @buf18, i64 %134
  %136 = load i32, ptr %135, align 4
  %137 = mul i32 %29, %136
  %138 = add i32 %133, %137
  %139 = or i64 %116, 4
  %140 = getelementptr i32, ptr @buf18, i64 %139
  %141 = load i32, ptr %140, align 4
  %142 = mul i32 %32, %141
  %143 = add i32 %138, %142
  %144 = or i64 %116, 5
  %145 = getelementptr i32, ptr @buf18, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %35, %146
  %148 = add i32 %143, %147
  %149 = or i64 %116, 6
  %150 = getelementptr i32, ptr @buf18, i64 %149
  %151 = load i32, ptr %150, align 4
  %152 = mul i32 %38, %151
  %153 = add i32 %148, %152
  %154 = or i64 %116, 7
  %155 = getelementptr i32, ptr @buf18, i64 %154
  %156 = load i32, ptr %155, align 4
  %157 = mul i32 %41, %156
  %158 = add i32 %153, %157
  store i32 %158, ptr %120, align 4
  %159 = add nuw nsw i64 %9, %117
  %160 = getelementptr i32, ptr @buf19, i64 %159
  %.promoted.1 = load i32, ptr %160, align 4
  %161 = mul i32 %44, %121
  %162 = add i32 %.promoted.1, %161
  %163 = mul i32 %47, %126
  %164 = add i32 %162, %163
  %165 = mul i32 %50, %131
  %166 = add i32 %164, %165
  %167 = mul i32 %53, %136
  %168 = add i32 %166, %167
  %169 = mul i32 %56, %141
  %170 = add i32 %168, %169
  %171 = mul i32 %59, %146
  %172 = add i32 %170, %171
  %173 = mul i32 %62, %151
  %174 = add i32 %172, %173
  %175 = mul i32 %65, %156
  %176 = add i32 %174, %175
  store i32 %176, ptr %160, align 4
  %177 = add nuw nsw i64 %11, %117
  %178 = getelementptr i32, ptr @buf19, i64 %177
  %.promoted.2 = load i32, ptr %178, align 4
  %179 = load i32, ptr %118, align 4
  %180 = mul i32 %68, %179
  %181 = add i32 %.promoted.2, %180
  %182 = load i32, ptr %125, align 4
  %183 = mul i32 %71, %182
  %184 = add i32 %181, %183
  %185 = load i32, ptr %130, align 4
  %186 = mul i32 %74, %185
  %187 = add i32 %184, %186
  %188 = load i32, ptr %135, align 4
  %189 = mul i32 %77, %188
  %190 = add i32 %187, %189
  %191 = load i32, ptr %140, align 4
  %192 = mul i32 %80, %191
  %193 = add i32 %190, %192
  %194 = load i32, ptr %145, align 4
  %195 = mul i32 %83, %194
  %196 = add i32 %193, %195
  %197 = load i32, ptr %150, align 4
  %198 = mul i32 %86, %197
  %199 = add i32 %196, %198
  %200 = load i32, ptr %155, align 4
  %201 = mul i32 %89, %200
  %202 = add i32 %199, %201
  store i32 %202, ptr %178, align 4
  %203 = add nuw nsw i64 %13, %117
  %204 = getelementptr i32, ptr @buf19, i64 %203
  %.promoted.3 = load i32, ptr %204, align 4
  %205 = mul i32 %92, %179
  %206 = add i32 %.promoted.3, %205
  %207 = mul i32 %95, %182
  %208 = add i32 %206, %207
  %209 = mul i32 %98, %185
  %210 = add i32 %208, %209
  %211 = mul i32 %101, %188
  %212 = add i32 %210, %211
  %213 = mul i32 %104, %191
  %214 = add i32 %212, %213
  %215 = mul i32 %107, %194
  %216 = add i32 %214, %215
  %217 = mul i32 %110, %197
  %218 = add i32 %216, %217
  %219 = mul i32 %113, %200
  %220 = add i32 %218, %219
  store i32 %220, ptr %204, align 4
  %221 = add nuw nsw i64 %114, 1
  %222 = icmp ult i64 %114, 3
  br i1 %222, label %.preheader5, label %223

223:                                              ; preds = %.preheader5
  %224 = add nuw nsw i64 %14, 1
  %225 = icmp ult i64 %14, 3
  br i1 %225, label %.preheader7, label %226

226:                                              ; preds = %223
  %227 = add nuw nsw i64 %5, 1
  %228 = icmp ult i64 %5, 7
  br i1 %228, label %.preheader10, label %229

229:                                              ; preds = %226
  %230 = add nuw nsw i64 %2, 1
  %231 = icmp ult i64 %2, 7
  br i1 %231, label %.preheader13, label %232

232:                                              ; preds = %229
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf16 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf15 to i64), i64 31), i64 0))
  tail call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint (ptr @buf19 to i64), i64 31), i64 0))
  br label %.preheader12

.preheader12:                                     ; preds = %232, %460
  %233 = phi i64 [ 0, %232 ], [ %461, %460 ]
  %234 = shl nuw nsw i64 %233, 5
  %235 = shl nuw nsw i64 %233, 4
  br label %.preheader9

.preheader9:                                      ; preds = %.preheader12, %457
  %236 = phi i64 [ 0, %.preheader12 ], [ %458, %457 ]
  %237 = shl nuw nsw i64 %236, 7
  %238 = add nuw nsw i64 %237, %235
  %239 = or i64 %237, 1
  %240 = add nuw nsw i64 %239, %235
  %241 = or i64 %237, 2
  %242 = add nuw nsw i64 %241, %235
  %243 = or i64 %237, 3
  %244 = add nuw nsw i64 %243, %235
  br label %.preheader6

.preheader6:                                      ; preds = %.preheader9, %454
  %245 = phi i64 [ 0, %.preheader9 ], [ %455, %454 ]
  %246 = shl nuw nsw i64 %245, 8
  %247 = add nuw nsw i64 %246, %234
  %248 = shl nuw nsw i64 %245, 5
  %249 = add nuw nsw i64 %237, %248
  %250 = getelementptr i32, ptr @buf15, i64 %249
  %251 = load i32, ptr %250, align 4
  %252 = or i64 %249, 4
  %253 = getelementptr i32, ptr @buf15, i64 %252
  %254 = load i32, ptr %253, align 4
  %255 = or i64 %249, 8
  %256 = getelementptr i32, ptr @buf15, i64 %255
  %257 = load i32, ptr %256, align 4
  %258 = or i64 %249, 12
  %259 = getelementptr i32, ptr @buf15, i64 %258
  %260 = load i32, ptr %259, align 4
  %261 = or i64 %249, 16
  %262 = getelementptr i32, ptr @buf15, i64 %261
  %263 = load i32, ptr %262, align 4
  %264 = or i64 %249, 20
  %265 = getelementptr i32, ptr @buf15, i64 %264
  %266 = load i32, ptr %265, align 4
  %267 = or i64 %249, 24
  %268 = getelementptr i32, ptr @buf15, i64 %267
  %269 = load i32, ptr %268, align 4
  %270 = or i64 %249, 28
  %271 = getelementptr i32, ptr @buf15, i64 %270
  %272 = load i32, ptr %271, align 4
  %273 = add nuw nsw i64 %239, %248
  %274 = getelementptr i32, ptr @buf15, i64 %273
  %275 = load i32, ptr %274, align 4
  %276 = or i64 %273, 4
  %277 = getelementptr i32, ptr @buf15, i64 %276
  %278 = load i32, ptr %277, align 4
  %279 = or i64 %273, 8
  %280 = getelementptr i32, ptr @buf15, i64 %279
  %281 = load i32, ptr %280, align 4
  %282 = or i64 %273, 12
  %283 = getelementptr i32, ptr @buf15, i64 %282
  %284 = load i32, ptr %283, align 4
  %285 = or i64 %273, 16
  %286 = getelementptr i32, ptr @buf15, i64 %285
  %287 = load i32, ptr %286, align 4
  %288 = or i64 %273, 20
  %289 = getelementptr i32, ptr @buf15, i64 %288
  %290 = load i32, ptr %289, align 4
  %291 = or i64 %273, 24
  %292 = getelementptr i32, ptr @buf15, i64 %291
  %293 = load i32, ptr %292, align 4
  %294 = or i64 %273, 28
  %295 = getelementptr i32, ptr @buf15, i64 %294
  %296 = load i32, ptr %295, align 4
  %297 = add nuw nsw i64 %241, %248
  %298 = getelementptr i32, ptr @buf15, i64 %297
  %299 = load i32, ptr %298, align 4
  %300 = or i64 %297, 4
  %301 = getelementptr i32, ptr @buf15, i64 %300
  %302 = load i32, ptr %301, align 4
  %303 = or i64 %297, 8
  %304 = getelementptr i32, ptr @buf15, i64 %303
  %305 = load i32, ptr %304, align 4
  %306 = or i64 %297, 12
  %307 = getelementptr i32, ptr @buf15, i64 %306
  %308 = load i32, ptr %307, align 4
  %309 = or i64 %297, 16
  %310 = getelementptr i32, ptr @buf15, i64 %309
  %311 = load i32, ptr %310, align 4
  %312 = or i64 %297, 20
  %313 = getelementptr i32, ptr @buf15, i64 %312
  %314 = load i32, ptr %313, align 4
  %315 = or i64 %297, 24
  %316 = getelementptr i32, ptr @buf15, i64 %315
  %317 = load i32, ptr %316, align 4
  %318 = or i64 %297, 28
  %319 = getelementptr i32, ptr @buf15, i64 %318
  %320 = load i32, ptr %319, align 4
  %321 = add nuw nsw i64 %243, %248
  %322 = getelementptr i32, ptr @buf15, i64 %321
  %323 = load i32, ptr %322, align 4
  %324 = or i64 %321, 4
  %325 = getelementptr i32, ptr @buf15, i64 %324
  %326 = load i32, ptr %325, align 4
  %327 = or i64 %321, 8
  %328 = getelementptr i32, ptr @buf15, i64 %327
  %329 = load i32, ptr %328, align 4
  %330 = or i64 %321, 12
  %331 = getelementptr i32, ptr @buf15, i64 %330
  %332 = load i32, ptr %331, align 4
  %333 = or i64 %321, 16
  %334 = getelementptr i32, ptr @buf15, i64 %333
  %335 = load i32, ptr %334, align 4
  %336 = or i64 %321, 20
  %337 = getelementptr i32, ptr @buf15, i64 %336
  %338 = load i32, ptr %337, align 4
  %339 = or i64 %321, 24
  %340 = getelementptr i32, ptr @buf15, i64 %339
  %341 = load i32, ptr %340, align 4
  %342 = or i64 %321, 28
  %343 = getelementptr i32, ptr @buf15, i64 %342
  %344 = load i32, ptr %343, align 4
  br label %.preheader4

.preheader4:                                      ; preds = %.preheader6, %.preheader4
  %345 = phi i64 [ 0, %.preheader6 ], [ %452, %.preheader4 ]
  %346 = shl nuw nsw i64 %345, 3
  %347 = add nuw nsw i64 %247, %346
  %348 = shl nuw nsw i64 %345, 2
  %349 = getelementptr i32, ptr @buf16, i64 %347
  %350 = add nuw nsw i64 %238, %348
  %351 = getelementptr i32, ptr @buf19, i64 %350
  %.promoted15 = load i32, ptr %351, align 4
  %352 = load i32, ptr %349, align 4
  %353 = mul i32 %251, %352
  %354 = add i32 %.promoted15, %353
  %355 = or i64 %347, 1
  %356 = getelementptr i32, ptr @buf16, i64 %355
  %357 = load i32, ptr %356, align 4
  %358 = mul i32 %254, %357
  %359 = add i32 %354, %358
  %360 = or i64 %347, 2
  %361 = getelementptr i32, ptr @buf16, i64 %360
  %362 = load i32, ptr %361, align 4
  %363 = mul i32 %257, %362
  %364 = add i32 %359, %363
  %365 = or i64 %347, 3
  %366 = getelementptr i32, ptr @buf16, i64 %365
  %367 = load i32, ptr %366, align 4
  %368 = mul i32 %260, %367
  %369 = add i32 %364, %368
  %370 = or i64 %347, 4
  %371 = getelementptr i32, ptr @buf16, i64 %370
  %372 = load i32, ptr %371, align 4
  %373 = mul i32 %263, %372
  %374 = add i32 %369, %373
  %375 = or i64 %347, 5
  %376 = getelementptr i32, ptr @buf16, i64 %375
  %377 = load i32, ptr %376, align 4
  %378 = mul i32 %266, %377
  %379 = add i32 %374, %378
  %380 = or i64 %347, 6
  %381 = getelementptr i32, ptr @buf16, i64 %380
  %382 = load i32, ptr %381, align 4
  %383 = mul i32 %269, %382
  %384 = add i32 %379, %383
  %385 = or i64 %347, 7
  %386 = getelementptr i32, ptr @buf16, i64 %385
  %387 = load i32, ptr %386, align 4
  %388 = mul i32 %272, %387
  %389 = add i32 %384, %388
  store i32 %389, ptr %351, align 4
  %390 = add nuw nsw i64 %240, %348
  %391 = getelementptr i32, ptr @buf19, i64 %390
  %.promoted15.1 = load i32, ptr %391, align 4
  %392 = mul i32 %275, %352
  %393 = add i32 %.promoted15.1, %392
  %394 = mul i32 %278, %357
  %395 = add i32 %393, %394
  %396 = mul i32 %281, %362
  %397 = add i32 %395, %396
  %398 = mul i32 %284, %367
  %399 = add i32 %397, %398
  %400 = mul i32 %287, %372
  %401 = add i32 %399, %400
  %402 = mul i32 %290, %377
  %403 = add i32 %401, %402
  %404 = mul i32 %293, %382
  %405 = add i32 %403, %404
  %406 = mul i32 %296, %387
  %407 = add i32 %405, %406
  store i32 %407, ptr %391, align 4
  %408 = add nuw nsw i64 %242, %348
  %409 = getelementptr i32, ptr @buf19, i64 %408
  %.promoted15.2 = load i32, ptr %409, align 4
  %410 = load i32, ptr %349, align 4
  %411 = mul i32 %299, %410
  %412 = add i32 %.promoted15.2, %411
  %413 = load i32, ptr %356, align 4
  %414 = mul i32 %302, %413
  %415 = add i32 %412, %414
  %416 = load i32, ptr %361, align 4
  %417 = mul i32 %305, %416
  %418 = add i32 %415, %417
  %419 = load i32, ptr %366, align 4
  %420 = mul i32 %308, %419
  %421 = add i32 %418, %420
  %422 = load i32, ptr %371, align 4
  %423 = mul i32 %311, %422
  %424 = add i32 %421, %423
  %425 = load i32, ptr %376, align 4
  %426 = mul i32 %314, %425
  %427 = add i32 %424, %426
  %428 = load i32, ptr %381, align 4
  %429 = mul i32 %317, %428
  %430 = add i32 %427, %429
  %431 = load i32, ptr %386, align 4
  %432 = mul i32 %320, %431
  %433 = add i32 %430, %432
  store i32 %433, ptr %409, align 4
  %434 = add nuw nsw i64 %244, %348
  %435 = getelementptr i32, ptr @buf19, i64 %434
  %.promoted15.3 = load i32, ptr %435, align 4
  %436 = mul i32 %323, %410
  %437 = add i32 %.promoted15.3, %436
  %438 = mul i32 %326, %413
  %439 = add i32 %437, %438
  %440 = mul i32 %329, %416
  %441 = add i32 %439, %440
  %442 = mul i32 %332, %419
  %443 = add i32 %441, %442
  %444 = mul i32 %335, %422
  %445 = add i32 %443, %444
  %446 = mul i32 %338, %425
  %447 = add i32 %445, %446
  %448 = mul i32 %341, %428
  %449 = add i32 %447, %448
  %450 = mul i32 %344, %431
  %451 = add i32 %449, %450
  store i32 %451, ptr %435, align 4
  %452 = add nuw nsw i64 %345, 1
  %453 = icmp ult i64 %345, 3
  br i1 %453, label %.preheader4, label %454

454:                                              ; preds = %.preheader4
  %455 = add nuw nsw i64 %245, 1
  %456 = icmp ult i64 %245, 3
  br i1 %456, label %.preheader6, label %457

457:                                              ; preds = %454
  %458 = add nuw nsw i64 %236, 1
  %459 = icmp ult i64 %236, 7
  br i1 %459, label %.preheader9, label %460

460:                                              ; preds = %457
  %461 = add nuw nsw i64 %233, 1
  %462 = icmp ult i64 %233, 7
  br i1 %462, label %.preheader12, label %463

463:                                              ; preds = %460
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

attributes #0 = { nounwind }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { noreturn nounwind }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
