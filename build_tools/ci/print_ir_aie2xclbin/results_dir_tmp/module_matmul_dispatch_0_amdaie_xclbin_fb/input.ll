; ModuleID = 'LLVMDialectModule'
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
@buf20 = external global [1 x [1 x [64 x [64 x i32]]]]
@buf21 = external global [1 x [1 x [64 x [64 x i32]]]]
@buf22 = external global [1 x [1 x [64 x [64 x i32]]]]
@airMemcpyId20 = external global [1 x [1 x [64 x [64 x i32]]]]
@airMemcpyId4 = external global [1 x [1 x [64 x [64 x i32]]]]
@airMemcpyId5 = external global [1 x [1 x [64 x [64 x i32]]]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @matmul_dispatch_0_matmul_64x64x64_i32(ptr %0, ptr %1, ptr %2) {
  %4 = ptrtoint ptr %0 to i64
  %5 = and i64 %4, 63
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = ptrtoint ptr %1 to i64
  %8 = and i64 %7, 63
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  %10 = ptrtoint ptr %2 to i64
  %11 = and i64 %10, 63
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %160, %0
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %2

2:                                                ; preds = %30, %1
  %3 = phi i64 [ %31, %30 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 8
  br i1 %4, label %5, label %32

5:                                                ; preds = %28, %2
  %6 = phi i64 [ %29, %28 ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %30

8:                                                ; preds = %26, %5
  %9 = phi i64 [ %27, %26 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %28

11:                                               ; preds = %14, %8
  %12 = phi i64 [ %25, %14 ], [ 0, %8 ]
  %13 = icmp slt i64 %12, 4
  br i1 %13, label %14, label %26

14:                                               ; preds = %11
  %15 = and i64 ptrtoint (ptr @buf4 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = mul i64 %3, 128
  %18 = add i64 0, %17
  %19 = mul i64 %6, 16
  %20 = add i64 %18, %19
  %21 = mul i64 %9, 4
  %22 = add i64 %20, %21
  %23 = add i64 %22, %12
  %24 = getelementptr i32, ptr @buf4, i64 %23
  store i32 0, ptr %24, align 4
  %25 = add i64 %12, 1
  br label %11

26:                                               ; preds = %11
  %27 = add i64 %9, 1
  br label %8

28:                                               ; preds = %8
  %29 = add i64 %6, 1
  br label %5

30:                                               ; preds = %5
  %31 = add i64 %3, 1
  br label %2

32:                                               ; preds = %2
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %33

33:                                               ; preds = %94, %32
  %34 = phi i64 [ %95, %94 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 8
  br i1 %35, label %36, label %96

36:                                               ; preds = %92, %33
  %37 = phi i64 [ %93, %92 ], [ 0, %33 ]
  %38 = icmp slt i64 %37, 8
  br i1 %38, label %39, label %94

39:                                               ; preds = %90, %36
  %40 = phi i64 [ %91, %90 ], [ 0, %36 ]
  %41 = icmp slt i64 %40, 4
  br i1 %41, label %42, label %92

42:                                               ; preds = %88, %39
  %43 = phi i64 [ %89, %88 ], [ 0, %39 ]
  %44 = icmp slt i64 %43, 4
  br i1 %44, label %45, label %90

45:                                               ; preds = %86, %42
  %46 = phi i64 [ %87, %86 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 4
  br i1 %47, label %48, label %88

48:                                               ; preds = %51, %45
  %49 = phi i64 [ %85, %51 ], [ 0, %45 ]
  %50 = icmp slt i64 %49, 8
  br i1 %50, label %51, label %86

51:                                               ; preds = %48
  %52 = and i64 ptrtoint (ptr @buf3 to i64), 31
  %53 = icmp eq i64 %52, 0
  call void @llvm.assume(i1 %53)
  %54 = mul i64 %40, 256
  %55 = add i64 0, %54
  %56 = mul i64 %34, 32
  %57 = add i64 %55, %56
  %58 = mul i64 %43, 8
  %59 = add i64 %57, %58
  %60 = add i64 %59, %49
  %61 = getelementptr i32, ptr @buf3, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = and i64 ptrtoint (ptr @buf2 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  %65 = mul i64 %37, 128
  %66 = add i64 0, %65
  %67 = mul i64 %40, 32
  %68 = add i64 %66, %67
  %69 = mul i64 %49, 4
  %70 = add i64 %68, %69
  %71 = add i64 %70, %46
  %72 = getelementptr i32, ptr @buf2, i64 %71
  %73 = load i32, ptr %72, align 4
  %74 = and i64 ptrtoint (ptr @buf4 to i64), 31
  %75 = icmp eq i64 %74, 0
  call void @llvm.assume(i1 %75)
  %76 = mul i64 %34, 16
  %77 = add i64 %66, %76
  %78 = mul i64 %43, 4
  %79 = add i64 %77, %78
  %80 = add i64 %79, %46
  %81 = getelementptr i32, ptr @buf4, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = mul i32 %62, %73
  %84 = add i32 %82, %83
  call void @llvm.assume(i1 %75)
  store i32 %84, ptr %81, align 4
  %85 = add i64 %49, 1
  br label %48

86:                                               ; preds = %48
  %87 = add i64 %46, 1
  br label %45

88:                                               ; preds = %45
  %89 = add i64 %43, 1
  br label %42

90:                                               ; preds = %42
  %91 = add i64 %40, 1
  br label %39

92:                                               ; preds = %39
  %93 = add i64 %37, 1
  br label %36

94:                                               ; preds = %36
  %95 = add i64 %34, 1
  br label %33

96:                                               ; preds = %33
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %97

97:                                               ; preds = %158, %96
  %98 = phi i64 [ %159, %158 ], [ 0, %96 ]
  %99 = icmp slt i64 %98, 8
  br i1 %99, label %100, label %160

100:                                              ; preds = %156, %97
  %101 = phi i64 [ %157, %156 ], [ 0, %97 ]
  %102 = icmp slt i64 %101, 8
  br i1 %102, label %103, label %158

103:                                              ; preds = %154, %100
  %104 = phi i64 [ %155, %154 ], [ 0, %100 ]
  %105 = icmp slt i64 %104, 4
  br i1 %105, label %106, label %156

106:                                              ; preds = %152, %103
  %107 = phi i64 [ %153, %152 ], [ 0, %103 ]
  %108 = icmp slt i64 %107, 4
  br i1 %108, label %109, label %154

109:                                              ; preds = %150, %106
  %110 = phi i64 [ %151, %150 ], [ 0, %106 ]
  %111 = icmp slt i64 %110, 4
  br i1 %111, label %112, label %152

112:                                              ; preds = %115, %109
  %113 = phi i64 [ %149, %115 ], [ 0, %109 ]
  %114 = icmp slt i64 %113, 8
  br i1 %114, label %115, label %150

115:                                              ; preds = %112
  %116 = and i64 ptrtoint (ptr @buf1 to i64), 31
  %117 = icmp eq i64 %116, 0
  call void @llvm.assume(i1 %117)
  %118 = mul i64 %104, 256
  %119 = add i64 0, %118
  %120 = mul i64 %98, 32
  %121 = add i64 %119, %120
  %122 = mul i64 %107, 8
  %123 = add i64 %121, %122
  %124 = add i64 %123, %113
  %125 = getelementptr i32, ptr @buf1, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = and i64 ptrtoint (ptr @buf0 to i64), 31
  %128 = icmp eq i64 %127, 0
  call void @llvm.assume(i1 %128)
  %129 = mul i64 %101, 128
  %130 = add i64 0, %129
  %131 = mul i64 %104, 32
  %132 = add i64 %130, %131
  %133 = mul i64 %113, 4
  %134 = add i64 %132, %133
  %135 = add i64 %134, %110
  %136 = getelementptr i32, ptr @buf0, i64 %135
  %137 = load i32, ptr %136, align 4
  %138 = and i64 ptrtoint (ptr @buf4 to i64), 31
  %139 = icmp eq i64 %138, 0
  call void @llvm.assume(i1 %139)
  %140 = mul i64 %98, 16
  %141 = add i64 %130, %140
  %142 = mul i64 %107, 4
  %143 = add i64 %141, %142
  %144 = add i64 %143, %110
  %145 = getelementptr i32, ptr @buf4, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %126, %137
  %148 = add i32 %146, %147
  call void @llvm.assume(i1 %139)
  store i32 %148, ptr %145, align 4
  %149 = add i64 %113, 1
  br label %112

150:                                              ; preds = %112
  %151 = add i64 %110, 1
  br label %109

152:                                              ; preds = %109
  %153 = add i64 %107, 1
  br label %106

154:                                              ; preds = %106
  %155 = add i64 %104, 1
  br label %103

156:                                              ; preds = %103
  %157 = add i64 %101, 1
  br label %100

158:                                              ; preds = %100
  %159 = add i64 %98, 1
  br label %97

160:                                              ; preds = %97
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

define void @core_0_3() {
  br label %1

1:                                                ; preds = %160, %0
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %2

2:                                                ; preds = %30, %1
  %3 = phi i64 [ %31, %30 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 8
  br i1 %4, label %5, label %32

5:                                                ; preds = %28, %2
  %6 = phi i64 [ %29, %28 ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %30

8:                                                ; preds = %26, %5
  %9 = phi i64 [ %27, %26 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %28

11:                                               ; preds = %14, %8
  %12 = phi i64 [ %25, %14 ], [ 0, %8 ]
  %13 = icmp slt i64 %12, 4
  br i1 %13, label %14, label %26

14:                                               ; preds = %11
  %15 = and i64 ptrtoint (ptr @buf9 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = mul i64 %3, 128
  %18 = add i64 0, %17
  %19 = mul i64 %6, 16
  %20 = add i64 %18, %19
  %21 = mul i64 %9, 4
  %22 = add i64 %20, %21
  %23 = add i64 %22, %12
  %24 = getelementptr i32, ptr @buf9, i64 %23
  store i32 0, ptr %24, align 4
  %25 = add i64 %12, 1
  br label %11

26:                                               ; preds = %11
  %27 = add i64 %9, 1
  br label %8

28:                                               ; preds = %8
  %29 = add i64 %6, 1
  br label %5

30:                                               ; preds = %5
  %31 = add i64 %3, 1
  br label %2

32:                                               ; preds = %2
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %33

33:                                               ; preds = %94, %32
  %34 = phi i64 [ %95, %94 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 8
  br i1 %35, label %36, label %96

36:                                               ; preds = %92, %33
  %37 = phi i64 [ %93, %92 ], [ 0, %33 ]
  %38 = icmp slt i64 %37, 8
  br i1 %38, label %39, label %94

39:                                               ; preds = %90, %36
  %40 = phi i64 [ %91, %90 ], [ 0, %36 ]
  %41 = icmp slt i64 %40, 4
  br i1 %41, label %42, label %92

42:                                               ; preds = %88, %39
  %43 = phi i64 [ %89, %88 ], [ 0, %39 ]
  %44 = icmp slt i64 %43, 4
  br i1 %44, label %45, label %90

45:                                               ; preds = %86, %42
  %46 = phi i64 [ %87, %86 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 4
  br i1 %47, label %48, label %88

48:                                               ; preds = %51, %45
  %49 = phi i64 [ %85, %51 ], [ 0, %45 ]
  %50 = icmp slt i64 %49, 8
  br i1 %50, label %51, label %86

51:                                               ; preds = %48
  %52 = and i64 ptrtoint (ptr @buf8 to i64), 31
  %53 = icmp eq i64 %52, 0
  call void @llvm.assume(i1 %53)
  %54 = mul i64 %40, 256
  %55 = add i64 0, %54
  %56 = mul i64 %34, 32
  %57 = add i64 %55, %56
  %58 = mul i64 %43, 8
  %59 = add i64 %57, %58
  %60 = add i64 %59, %49
  %61 = getelementptr i32, ptr @buf8, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = and i64 ptrtoint (ptr @buf7 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  %65 = mul i64 %37, 128
  %66 = add i64 0, %65
  %67 = mul i64 %40, 32
  %68 = add i64 %66, %67
  %69 = mul i64 %49, 4
  %70 = add i64 %68, %69
  %71 = add i64 %70, %46
  %72 = getelementptr i32, ptr @buf7, i64 %71
  %73 = load i32, ptr %72, align 4
  %74 = and i64 ptrtoint (ptr @buf9 to i64), 31
  %75 = icmp eq i64 %74, 0
  call void @llvm.assume(i1 %75)
  %76 = mul i64 %34, 16
  %77 = add i64 %66, %76
  %78 = mul i64 %43, 4
  %79 = add i64 %77, %78
  %80 = add i64 %79, %46
  %81 = getelementptr i32, ptr @buf9, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = mul i32 %62, %73
  %84 = add i32 %82, %83
  call void @llvm.assume(i1 %75)
  store i32 %84, ptr %81, align 4
  %85 = add i64 %49, 1
  br label %48

86:                                               ; preds = %48
  %87 = add i64 %46, 1
  br label %45

88:                                               ; preds = %45
  %89 = add i64 %43, 1
  br label %42

90:                                               ; preds = %42
  %91 = add i64 %40, 1
  br label %39

92:                                               ; preds = %39
  %93 = add i64 %37, 1
  br label %36

94:                                               ; preds = %36
  %95 = add i64 %34, 1
  br label %33

96:                                               ; preds = %33
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %97

97:                                               ; preds = %158, %96
  %98 = phi i64 [ %159, %158 ], [ 0, %96 ]
  %99 = icmp slt i64 %98, 8
  br i1 %99, label %100, label %160

100:                                              ; preds = %156, %97
  %101 = phi i64 [ %157, %156 ], [ 0, %97 ]
  %102 = icmp slt i64 %101, 8
  br i1 %102, label %103, label %158

103:                                              ; preds = %154, %100
  %104 = phi i64 [ %155, %154 ], [ 0, %100 ]
  %105 = icmp slt i64 %104, 4
  br i1 %105, label %106, label %156

106:                                              ; preds = %152, %103
  %107 = phi i64 [ %153, %152 ], [ 0, %103 ]
  %108 = icmp slt i64 %107, 4
  br i1 %108, label %109, label %154

109:                                              ; preds = %150, %106
  %110 = phi i64 [ %151, %150 ], [ 0, %106 ]
  %111 = icmp slt i64 %110, 4
  br i1 %111, label %112, label %152

112:                                              ; preds = %115, %109
  %113 = phi i64 [ %149, %115 ], [ 0, %109 ]
  %114 = icmp slt i64 %113, 8
  br i1 %114, label %115, label %150

115:                                              ; preds = %112
  %116 = and i64 ptrtoint (ptr @buf6 to i64), 31
  %117 = icmp eq i64 %116, 0
  call void @llvm.assume(i1 %117)
  %118 = mul i64 %104, 256
  %119 = add i64 0, %118
  %120 = mul i64 %98, 32
  %121 = add i64 %119, %120
  %122 = mul i64 %107, 8
  %123 = add i64 %121, %122
  %124 = add i64 %123, %113
  %125 = getelementptr i32, ptr @buf6, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = and i64 ptrtoint (ptr @buf5 to i64), 31
  %128 = icmp eq i64 %127, 0
  call void @llvm.assume(i1 %128)
  %129 = mul i64 %101, 128
  %130 = add i64 0, %129
  %131 = mul i64 %104, 32
  %132 = add i64 %130, %131
  %133 = mul i64 %113, 4
  %134 = add i64 %132, %133
  %135 = add i64 %134, %110
  %136 = getelementptr i32, ptr @buf5, i64 %135
  %137 = load i32, ptr %136, align 4
  %138 = and i64 ptrtoint (ptr @buf9 to i64), 31
  %139 = icmp eq i64 %138, 0
  call void @llvm.assume(i1 %139)
  %140 = mul i64 %98, 16
  %141 = add i64 %130, %140
  %142 = mul i64 %107, 4
  %143 = add i64 %141, %142
  %144 = add i64 %143, %110
  %145 = getelementptr i32, ptr @buf9, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %126, %137
  %148 = add i32 %146, %147
  call void @llvm.assume(i1 %139)
  store i32 %148, ptr %145, align 4
  %149 = add i64 %113, 1
  br label %112

150:                                              ; preds = %112
  %151 = add i64 %110, 1
  br label %109

152:                                              ; preds = %109
  %153 = add i64 %107, 1
  br label %106

154:                                              ; preds = %106
  %155 = add i64 %104, 1
  br label %103

156:                                              ; preds = %103
  %157 = add i64 %101, 1
  br label %100

158:                                              ; preds = %100
  %159 = add i64 %98, 1
  br label %97

160:                                              ; preds = %97
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

define void @core_0_4() {
  br label %1

1:                                                ; preds = %160, %0
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %2

2:                                                ; preds = %30, %1
  %3 = phi i64 [ %31, %30 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 8
  br i1 %4, label %5, label %32

5:                                                ; preds = %28, %2
  %6 = phi i64 [ %29, %28 ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %30

8:                                                ; preds = %26, %5
  %9 = phi i64 [ %27, %26 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %28

11:                                               ; preds = %14, %8
  %12 = phi i64 [ %25, %14 ], [ 0, %8 ]
  %13 = icmp slt i64 %12, 4
  br i1 %13, label %14, label %26

14:                                               ; preds = %11
  %15 = and i64 ptrtoint (ptr @buf14 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = mul i64 %3, 128
  %18 = add i64 0, %17
  %19 = mul i64 %6, 16
  %20 = add i64 %18, %19
  %21 = mul i64 %9, 4
  %22 = add i64 %20, %21
  %23 = add i64 %22, %12
  %24 = getelementptr i32, ptr @buf14, i64 %23
  store i32 0, ptr %24, align 4
  %25 = add i64 %12, 1
  br label %11

26:                                               ; preds = %11
  %27 = add i64 %9, 1
  br label %8

28:                                               ; preds = %8
  %29 = add i64 %6, 1
  br label %5

30:                                               ; preds = %5
  %31 = add i64 %3, 1
  br label %2

32:                                               ; preds = %2
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %33

33:                                               ; preds = %94, %32
  %34 = phi i64 [ %95, %94 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 8
  br i1 %35, label %36, label %96

36:                                               ; preds = %92, %33
  %37 = phi i64 [ %93, %92 ], [ 0, %33 ]
  %38 = icmp slt i64 %37, 8
  br i1 %38, label %39, label %94

39:                                               ; preds = %90, %36
  %40 = phi i64 [ %91, %90 ], [ 0, %36 ]
  %41 = icmp slt i64 %40, 4
  br i1 %41, label %42, label %92

42:                                               ; preds = %88, %39
  %43 = phi i64 [ %89, %88 ], [ 0, %39 ]
  %44 = icmp slt i64 %43, 4
  br i1 %44, label %45, label %90

45:                                               ; preds = %86, %42
  %46 = phi i64 [ %87, %86 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 4
  br i1 %47, label %48, label %88

48:                                               ; preds = %51, %45
  %49 = phi i64 [ %85, %51 ], [ 0, %45 ]
  %50 = icmp slt i64 %49, 8
  br i1 %50, label %51, label %86

51:                                               ; preds = %48
  %52 = and i64 ptrtoint (ptr @buf13 to i64), 31
  %53 = icmp eq i64 %52, 0
  call void @llvm.assume(i1 %53)
  %54 = mul i64 %40, 256
  %55 = add i64 0, %54
  %56 = mul i64 %34, 32
  %57 = add i64 %55, %56
  %58 = mul i64 %43, 8
  %59 = add i64 %57, %58
  %60 = add i64 %59, %49
  %61 = getelementptr i32, ptr @buf13, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = and i64 ptrtoint (ptr @buf12 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  %65 = mul i64 %37, 128
  %66 = add i64 0, %65
  %67 = mul i64 %40, 32
  %68 = add i64 %66, %67
  %69 = mul i64 %49, 4
  %70 = add i64 %68, %69
  %71 = add i64 %70, %46
  %72 = getelementptr i32, ptr @buf12, i64 %71
  %73 = load i32, ptr %72, align 4
  %74 = and i64 ptrtoint (ptr @buf14 to i64), 31
  %75 = icmp eq i64 %74, 0
  call void @llvm.assume(i1 %75)
  %76 = mul i64 %34, 16
  %77 = add i64 %66, %76
  %78 = mul i64 %43, 4
  %79 = add i64 %77, %78
  %80 = add i64 %79, %46
  %81 = getelementptr i32, ptr @buf14, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = mul i32 %62, %73
  %84 = add i32 %82, %83
  call void @llvm.assume(i1 %75)
  store i32 %84, ptr %81, align 4
  %85 = add i64 %49, 1
  br label %48

86:                                               ; preds = %48
  %87 = add i64 %46, 1
  br label %45

88:                                               ; preds = %45
  %89 = add i64 %43, 1
  br label %42

90:                                               ; preds = %42
  %91 = add i64 %40, 1
  br label %39

92:                                               ; preds = %39
  %93 = add i64 %37, 1
  br label %36

94:                                               ; preds = %36
  %95 = add i64 %34, 1
  br label %33

96:                                               ; preds = %33
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %97

97:                                               ; preds = %158, %96
  %98 = phi i64 [ %159, %158 ], [ 0, %96 ]
  %99 = icmp slt i64 %98, 8
  br i1 %99, label %100, label %160

100:                                              ; preds = %156, %97
  %101 = phi i64 [ %157, %156 ], [ 0, %97 ]
  %102 = icmp slt i64 %101, 8
  br i1 %102, label %103, label %158

103:                                              ; preds = %154, %100
  %104 = phi i64 [ %155, %154 ], [ 0, %100 ]
  %105 = icmp slt i64 %104, 4
  br i1 %105, label %106, label %156

106:                                              ; preds = %152, %103
  %107 = phi i64 [ %153, %152 ], [ 0, %103 ]
  %108 = icmp slt i64 %107, 4
  br i1 %108, label %109, label %154

109:                                              ; preds = %150, %106
  %110 = phi i64 [ %151, %150 ], [ 0, %106 ]
  %111 = icmp slt i64 %110, 4
  br i1 %111, label %112, label %152

112:                                              ; preds = %115, %109
  %113 = phi i64 [ %149, %115 ], [ 0, %109 ]
  %114 = icmp slt i64 %113, 8
  br i1 %114, label %115, label %150

115:                                              ; preds = %112
  %116 = and i64 ptrtoint (ptr @buf11 to i64), 31
  %117 = icmp eq i64 %116, 0
  call void @llvm.assume(i1 %117)
  %118 = mul i64 %104, 256
  %119 = add i64 0, %118
  %120 = mul i64 %98, 32
  %121 = add i64 %119, %120
  %122 = mul i64 %107, 8
  %123 = add i64 %121, %122
  %124 = add i64 %123, %113
  %125 = getelementptr i32, ptr @buf11, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = and i64 ptrtoint (ptr @buf10 to i64), 31
  %128 = icmp eq i64 %127, 0
  call void @llvm.assume(i1 %128)
  %129 = mul i64 %101, 128
  %130 = add i64 0, %129
  %131 = mul i64 %104, 32
  %132 = add i64 %130, %131
  %133 = mul i64 %113, 4
  %134 = add i64 %132, %133
  %135 = add i64 %134, %110
  %136 = getelementptr i32, ptr @buf10, i64 %135
  %137 = load i32, ptr %136, align 4
  %138 = and i64 ptrtoint (ptr @buf14 to i64), 31
  %139 = icmp eq i64 %138, 0
  call void @llvm.assume(i1 %139)
  %140 = mul i64 %98, 16
  %141 = add i64 %130, %140
  %142 = mul i64 %107, 4
  %143 = add i64 %141, %142
  %144 = add i64 %143, %110
  %145 = getelementptr i32, ptr @buf14, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %126, %137
  %148 = add i32 %146, %147
  call void @llvm.assume(i1 %139)
  store i32 %148, ptr %145, align 4
  %149 = add i64 %113, 1
  br label %112

150:                                              ; preds = %112
  %151 = add i64 %110, 1
  br label %109

152:                                              ; preds = %109
  %153 = add i64 %107, 1
  br label %106

154:                                              ; preds = %106
  %155 = add i64 %104, 1
  br label %103

156:                                              ; preds = %103
  %157 = add i64 %101, 1
  br label %100

158:                                              ; preds = %100
  %159 = add i64 %98, 1
  br label %97

160:                                              ; preds = %97
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

define void @core_0_5() {
  br label %1

1:                                                ; preds = %160, %0
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %2

2:                                                ; preds = %30, %1
  %3 = phi i64 [ %31, %30 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 8
  br i1 %4, label %5, label %32

5:                                                ; preds = %28, %2
  %6 = phi i64 [ %29, %28 ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %30

8:                                                ; preds = %26, %5
  %9 = phi i64 [ %27, %26 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %28

11:                                               ; preds = %14, %8
  %12 = phi i64 [ %25, %14 ], [ 0, %8 ]
  %13 = icmp slt i64 %12, 4
  br i1 %13, label %14, label %26

14:                                               ; preds = %11
  %15 = and i64 ptrtoint (ptr @buf19 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = mul i64 %3, 128
  %18 = add i64 0, %17
  %19 = mul i64 %6, 16
  %20 = add i64 %18, %19
  %21 = mul i64 %9, 4
  %22 = add i64 %20, %21
  %23 = add i64 %22, %12
  %24 = getelementptr i32, ptr @buf19, i64 %23
  store i32 0, ptr %24, align 4
  %25 = add i64 %12, 1
  br label %11

26:                                               ; preds = %11
  %27 = add i64 %9, 1
  br label %8

28:                                               ; preds = %8
  %29 = add i64 %6, 1
  br label %5

30:                                               ; preds = %5
  %31 = add i64 %3, 1
  br label %2

32:                                               ; preds = %2
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %33

33:                                               ; preds = %94, %32
  %34 = phi i64 [ %95, %94 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 8
  br i1 %35, label %36, label %96

36:                                               ; preds = %92, %33
  %37 = phi i64 [ %93, %92 ], [ 0, %33 ]
  %38 = icmp slt i64 %37, 8
  br i1 %38, label %39, label %94

39:                                               ; preds = %90, %36
  %40 = phi i64 [ %91, %90 ], [ 0, %36 ]
  %41 = icmp slt i64 %40, 4
  br i1 %41, label %42, label %92

42:                                               ; preds = %88, %39
  %43 = phi i64 [ %89, %88 ], [ 0, %39 ]
  %44 = icmp slt i64 %43, 4
  br i1 %44, label %45, label %90

45:                                               ; preds = %86, %42
  %46 = phi i64 [ %87, %86 ], [ 0, %42 ]
  %47 = icmp slt i64 %46, 4
  br i1 %47, label %48, label %88

48:                                               ; preds = %51, %45
  %49 = phi i64 [ %85, %51 ], [ 0, %45 ]
  %50 = icmp slt i64 %49, 8
  br i1 %50, label %51, label %86

51:                                               ; preds = %48
  %52 = and i64 ptrtoint (ptr @buf18 to i64), 31
  %53 = icmp eq i64 %52, 0
  call void @llvm.assume(i1 %53)
  %54 = mul i64 %40, 256
  %55 = add i64 0, %54
  %56 = mul i64 %34, 32
  %57 = add i64 %55, %56
  %58 = mul i64 %43, 8
  %59 = add i64 %57, %58
  %60 = add i64 %59, %49
  %61 = getelementptr i32, ptr @buf18, i64 %60
  %62 = load i32, ptr %61, align 4
  %63 = and i64 ptrtoint (ptr @buf17 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  %65 = mul i64 %37, 128
  %66 = add i64 0, %65
  %67 = mul i64 %40, 32
  %68 = add i64 %66, %67
  %69 = mul i64 %49, 4
  %70 = add i64 %68, %69
  %71 = add i64 %70, %46
  %72 = getelementptr i32, ptr @buf17, i64 %71
  %73 = load i32, ptr %72, align 4
  %74 = and i64 ptrtoint (ptr @buf19 to i64), 31
  %75 = icmp eq i64 %74, 0
  call void @llvm.assume(i1 %75)
  %76 = mul i64 %34, 16
  %77 = add i64 %66, %76
  %78 = mul i64 %43, 4
  %79 = add i64 %77, %78
  %80 = add i64 %79, %46
  %81 = getelementptr i32, ptr @buf19, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = mul i32 %62, %73
  %84 = add i32 %82, %83
  call void @llvm.assume(i1 %75)
  store i32 %84, ptr %81, align 4
  %85 = add i64 %49, 1
  br label %48

86:                                               ; preds = %48
  %87 = add i64 %46, 1
  br label %45

88:                                               ; preds = %45
  %89 = add i64 %43, 1
  br label %42

90:                                               ; preds = %42
  %91 = add i64 %40, 1
  br label %39

92:                                               ; preds = %39
  %93 = add i64 %37, 1
  br label %36

94:                                               ; preds = %36
  %95 = add i64 %34, 1
  br label %33

96:                                               ; preds = %33
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  br label %97

97:                                               ; preds = %158, %96
  %98 = phi i64 [ %159, %158 ], [ 0, %96 ]
  %99 = icmp slt i64 %98, 8
  br i1 %99, label %100, label %160

100:                                              ; preds = %156, %97
  %101 = phi i64 [ %157, %156 ], [ 0, %97 ]
  %102 = icmp slt i64 %101, 8
  br i1 %102, label %103, label %158

103:                                              ; preds = %154, %100
  %104 = phi i64 [ %155, %154 ], [ 0, %100 ]
  %105 = icmp slt i64 %104, 4
  br i1 %105, label %106, label %156

106:                                              ; preds = %152, %103
  %107 = phi i64 [ %153, %152 ], [ 0, %103 ]
  %108 = icmp slt i64 %107, 4
  br i1 %108, label %109, label %154

109:                                              ; preds = %150, %106
  %110 = phi i64 [ %151, %150 ], [ 0, %106 ]
  %111 = icmp slt i64 %110, 4
  br i1 %111, label %112, label %152

112:                                              ; preds = %115, %109
  %113 = phi i64 [ %149, %115 ], [ 0, %109 ]
  %114 = icmp slt i64 %113, 8
  br i1 %114, label %115, label %150

115:                                              ; preds = %112
  %116 = and i64 ptrtoint (ptr @buf16 to i64), 31
  %117 = icmp eq i64 %116, 0
  call void @llvm.assume(i1 %117)
  %118 = mul i64 %104, 256
  %119 = add i64 0, %118
  %120 = mul i64 %98, 32
  %121 = add i64 %119, %120
  %122 = mul i64 %107, 8
  %123 = add i64 %121, %122
  %124 = add i64 %123, %113
  %125 = getelementptr i32, ptr @buf16, i64 %124
  %126 = load i32, ptr %125, align 4
  %127 = and i64 ptrtoint (ptr @buf15 to i64), 31
  %128 = icmp eq i64 %127, 0
  call void @llvm.assume(i1 %128)
  %129 = mul i64 %101, 128
  %130 = add i64 0, %129
  %131 = mul i64 %104, 32
  %132 = add i64 %130, %131
  %133 = mul i64 %113, 4
  %134 = add i64 %132, %133
  %135 = add i64 %134, %110
  %136 = getelementptr i32, ptr @buf15, i64 %135
  %137 = load i32, ptr %136, align 4
  %138 = and i64 ptrtoint (ptr @buf19 to i64), 31
  %139 = icmp eq i64 %138, 0
  call void @llvm.assume(i1 %139)
  %140 = mul i64 %98, 16
  %141 = add i64 %130, %140
  %142 = mul i64 %107, 4
  %143 = add i64 %141, %142
  %144 = add i64 %143, %110
  %145 = getelementptr i32, ptr @buf19, i64 %144
  %146 = load i32, ptr %145, align 4
  %147 = mul i32 %126, %137
  %148 = add i32 %146, %147
  call void @llvm.assume(i1 %139)
  store i32 %148, ptr %145, align 4
  %149 = add i64 %113, 1
  br label %112

150:                                              ; preds = %112
  %151 = add i64 %110, 1
  br label %109

152:                                              ; preds = %109
  %153 = add i64 %107, 1
  br label %106

154:                                              ; preds = %106
  %155 = add i64 %104, 1
  br label %103

156:                                              ; preds = %103
  %157 = add i64 %101, 1
  br label %100

158:                                              ; preds = %100
  %159 = add i64 %98, 1
  br label %97

160:                                              ; preds = %97
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
