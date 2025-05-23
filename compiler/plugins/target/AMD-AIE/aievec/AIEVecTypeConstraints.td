//===- AIEVecTypeConstraints.td - AIEVec type constraints--*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 AMD Inc.
//

//===----------------------------------------------------------------------===//
// Extra type constraint definitions for AIEVec operations.
//===----------------------------------------------------------------------===//
#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECTYPECONSTRAINTS_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECTYPECONSTRAINTS_TD

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def I4  : I<4>;
def I256 : I<256>;
def I512 : I<512>;

class TypeShape<string name> :
  StrFunc<"cast<::mlir::ShapedType>($" # name # ").getShape()">;

// Notice: This predicate class assumes that the type has been verified to be a
//         ShapedType
class VectorOfShape<list<int> shape> :
  CPred<TypeShape<"_self">.result # " == ArrayRef<int64_t>({" #
        !interleave(shape, ", ") # "})">;

// Notice: This predicate class assumes that the type has been verified to be a
//         ShapedType
class VectorOfElementType<Type type> :
  SubstLeaves<"$_self", ElementType<"_self">.result, type.predicate>;

// Notice: This predicate class assumes that the type has been verified to be a
//         ShapedType
class VectorOfShapeAndType<list<int> shape, Type type> :
  Type<And<[VectorOfShape<shape>,
            VectorOfElementType<type>]>,
        "vector of shape <" # !interleave(shape, "x") # "> and",
        "::mlir::VectorType">;

class ShapedTypeBitWidth<string name> :
  StrFunc<"(cast<::mlir::ShapedType>($" # name # ").getElementTypeBitWidth() * "
          # !subst(".getType()", "", ElementCount<name>.result) # ")">;

class VectorOfBitWidthAndElementTypes<int bitwidth, list<Type> allowedTypes> :
  Type<And<[VectorOfNonZeroRankOf<allowedTypes>.predicate,
            CPred<ShapedTypeBitWidth<"_self">.result # " == " # bitwidth>]>,
       bitwidth # "-bit wide vector, of " # AnyTypeOf<allowedTypes>.summary>;


class ShapeDimsMatch<string lhs, int ld, string rhs, int rd> :
  CPred<Shape<lhs>.result # "[" # ld # "] == " #
        Shape<rhs>.result # "[" # rd # "]">;

class VectorType<string name> : StrFunc<"cast<VectorType>($" # name #
                                        ".getType())">;

class VectorTypesMatch<string op1, Type t1,
                       string op2, Type t2,
                       string op3, Type t3> :
  And<[SubstLeaves<"$_self", VectorType<op1>.result, t1.predicate>,
       SubstLeaves<"$_self", VectorType<op2>.result, t2.predicate>,
       SubstLeaves<"$_self", VectorType<op3>.result, t3.predicate>]>;

#endif
