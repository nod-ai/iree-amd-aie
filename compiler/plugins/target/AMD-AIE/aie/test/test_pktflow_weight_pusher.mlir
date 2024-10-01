// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.masterset(DMA : 1, %[[VAL_0]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(31, 0, %[[VAL_0]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_4:.*]] = aie.masterset(DMA : 1, %[[VAL_3]])
// CHECK:             %[[VAL_5:.*]] = aie.masterset(WEST : 2, %[[VAL_2]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(31, 0, %[[VAL_2]])
// CHECK:               aie.rule(31, 4, %[[VAL_3]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_8:.*]] = aie.masterset(DMA : 1, %[[VAL_6]])
// CHECK:             %[[VAL_9:.*]] = aie.masterset(WEST : 2, %[[VAL_7]])
// CHECK:             aie.packet_rules(EAST : 0) {
// CHECK:               aie.rule(31, 8, %[[VAL_6]])
// CHECK:             }
// CHECK:             aie.packet_rules(EAST : 3) {
// CHECK:               aie.rule(27, 0, %[[VAL_7]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             %[[VAL_10:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_11:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_12:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_13:.*]] = aie.masterset(DMA : 1, %[[VAL_11]])
// CHECK:             %[[VAL_14:.*]] = aie.masterset(WEST : 0, %[[VAL_10]])
// CHECK:             %[[VAL_15:.*]] = aie.masterset(WEST : 3, %[[VAL_12]])
// CHECK:             aie.packet_rules(NORTH : 1) {
// CHECK:               aie.rule(31, 8, %[[VAL_10]])
// CHECK:               aie.rule(31, 12, %[[VAL_11]])
// CHECK:             }
// CHECK:             aie.packet_rules(EAST : 3) {
// CHECK:               aie.rule(27, 0, %[[VAL_12]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             %[[VAL_16:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_17:.*]] = aie.masterset(DMA : 1, %[[VAL_16]])
// CHECK:             aie.packet_rules(NORTH : 2) {
// CHECK:               aie.rule(31, 1, %[[VAL_16]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             %[[VAL_18:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_19:.*]] = aie.masterset(DMA : 1, %[[VAL_18]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(31, 5, %[[VAL_18]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             %[[VAL_20:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_21:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_22:.*]] = aie.masterset(DMA : 1, %[[VAL_20]])
// CHECK:             %[[VAL_23:.*]] = aie.masterset(WEST : 2, %[[VAL_21]])
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 9, %[[VAL_20]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 1) {
// CHECK:               aie.rule(31, 5, %[[VAL_21]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             %[[VAL_24:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_25:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_26:.*]] = aie.masterset(DMA : 1, %[[VAL_25]])
// CHECK:             %[[VAL_27:.*]] = aie.masterset(SOUTH : 1, %[[VAL_24]])
// CHECK:             aie.packet_rules(NORTH : 2) {
// CHECK:               aie.rule(27, 8, %[[VAL_24]])
// CHECK:               aie.rule(31, 13, %[[VAL_25]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_2_4:.*]] = aie.tile(2, 4)
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             %[[VAL_28:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_29:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_30:.*]] = aie.masterset(DMA : 1, %[[VAL_29]])
// CHECK:             %[[VAL_31:.*]] = aie.masterset(SOUTH : 2, %[[VAL_28]])
// CHECK:             aie.packet_rules(EAST : 1) {
// CHECK:               aie.rule(31, 1, %[[VAL_28]])
// CHECK:               aie.rule(31, 2, %[[VAL_29]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             %[[VAL_32:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_33:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_34:.*]] = aie.masterset(DMA : 1, %[[VAL_33]])
// CHECK:             %[[VAL_35:.*]] = aie.masterset(WEST : 1, %[[VAL_32]])
// CHECK:             aie.packet_rules(NORTH : 1) {
// CHECK:               aie.rule(28, 0, %[[VAL_32]])
// CHECK:               aie.rule(31, 6, %[[VAL_33]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             %[[VAL_36:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_37:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_38:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_39:.*]] = aie.masterset(DMA : 1, %[[VAL_37]])
// CHECK:             %[[VAL_40:.*]] = aie.masterset(SOUTH : 0, %[[VAL_36]])
// CHECK:             %[[VAL_41:.*]] = aie.masterset(SOUTH : 1, %[[VAL_38]])
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 9, %[[VAL_36]])
// CHECK:               aie.rule(31, 10, %[[VAL_37]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 3) {
// CHECK:               aie.rule(31, 5, %[[VAL_38]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:             %[[VAL_42:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_43:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_44:.*]] = aie.masterset(DMA : 1, %[[VAL_43]])
// CHECK:             %[[VAL_45:.*]] = aie.masterset(SOUTH : 2, %[[VAL_42]])
// CHECK:             aie.packet_rules(EAST : 1) {
// CHECK:               aie.rule(26, 8, %[[VAL_42]])
// CHECK:               aie.rule(31, 14, %[[VAL_43]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_2_5:.*]] = aie.tile(2, 5)
// CHECK:           %[[SWITCHBOX_2_5:.*]] = aie.switchbox(%[[TILE_2_5]]) {
// CHECK:             %[[VAL_46:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_47:.*]] = aie.masterset(DMA : 1, %[[VAL_46]])
// CHECK:             aie.packet_rules(EAST : 1) {
// CHECK:               aie.rule(31, 3, %[[VAL_46]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_3_5:.*]] = aie.tile(3, 5)
// CHECK:           %[[SWITCHBOX_3_5:.*]] = aie.switchbox(%[[TILE_3_5]]) {
// CHECK:             %[[VAL_48:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_49:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_50:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_51:.*]] = aie.masterset(DMA : 1, %[[VAL_50]])
// CHECK:             %[[VAL_52:.*]] = aie.masterset(SOUTH : 1, %[[VAL_48]])
// CHECK:             %[[VAL_53:.*]] = aie.masterset(WEST : 1, %[[VAL_49]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(24, 0, %[[VAL_48]])
// CHECK:               aie.rule(31, 3, %[[VAL_49]])
// CHECK:               aie.rule(31, 7, %[[VAL_50]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_4_5:.*]] = aie.tile(4, 5)
// CHECK:           %[[SWITCHBOX_4_5:.*]] = aie.switchbox(%[[TILE_4_5]]) {
// CHECK:             %[[VAL_54:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_55:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_56:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_57:.*]] = aie.amsel<3> (0)
// CHECK:             %[[VAL_58:.*]] = aie.masterset(DMA : 1, %[[VAL_57]])
// CHECK:             %[[VAL_59:.*]] = aie.masterset(SOUTH : 0, %[[VAL_56]])
// CHECK:             %[[VAL_60:.*]] = aie.masterset(SOUTH : 3, %[[VAL_55]])
// CHECK:             %[[VAL_61:.*]] = aie.masterset(WEST : 2, %[[VAL_54]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(24, 0, %[[VAL_54]])
// CHECK:               aie.rule(31, 5, %[[VAL_55]])
// CHECK:             }
// CHECK:             aie.packet_rules(EAST : 3) {
// CHECK:               aie.rule(28, 8, %[[VAL_56]])
// CHECK:               aie.rule(31, 11, %[[VAL_57]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_5_5:.*]] = aie.tile(5, 5)
// CHECK:           %[[SWITCHBOX_5_5:.*]] = aie.switchbox(%[[TILE_5_5]]) {
// CHECK:             %[[VAL_62:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_63:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_64:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_65:.*]] = aie.masterset(DMA : 1, %[[VAL_64]])
// CHECK:             %[[VAL_66:.*]] = aie.masterset(WEST : 2, %[[VAL_62]])
// CHECK:             %[[VAL_67:.*]] = aie.masterset(WEST : 3, %[[VAL_63]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(24, 0, %[[VAL_62]])
// CHECK:             }
// CHECK:             aie.packet_rules(EAST : 3) {
// CHECK:               aie.rule(28, 8, %[[VAL_63]])
// CHECK:               aie.rule(31, 15, %[[VAL_64]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_6_5:.*]] = aie.tile(6, 5)
// CHECK:           %[[SWITCHBOX_6_5:.*]] = aie.switchbox(%[[TILE_6_5]]) {
// CHECK:             %[[VAL_68:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_69:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_70:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_71:.*]] = aie.amsel<3> (0)
// CHECK:             %[[VAL_72:.*]] = aie.masterset(SOUTH : 2, %[[VAL_68]])
// CHECK:             %[[VAL_73:.*]] = aie.masterset(SOUTH : 3, %[[VAL_70]])
// CHECK:             %[[VAL_74:.*]] = aie.masterset(WEST : 2, %[[VAL_69]])
// CHECK:             %[[VAL_75:.*]] = aie.masterset(WEST : 3, %[[VAL_71]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(27, 0, %[[VAL_68]])
// CHECK:               aie.rule(24, 0, %[[VAL_69]])
// CHECK:             }
// CHECK:             aie.packet_rules(EAST : 0) {
// CHECK:               aie.rule(24, 8, %[[VAL_70]])
// CHECK:               aie.rule(24, 8, %[[VAL_71]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_7_5:.*]] = aie.tile(7, 5)
// CHECK:           %[[SWITCHBOX_7_5:.*]] = aie.switchbox(%[[TILE_7_5]]) {
// CHECK:             %[[VAL_76:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_77:.*]] = aie.masterset(WEST : 0, %[[VAL_76]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(24, 8, %[[VAL_76]])
// CHECK:             }
// CHECK:           }
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_2_2]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(1) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_2_3]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(2) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_2_4]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(3) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_2_5]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(4) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_3_2]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(5) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_3_3]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(6) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_3_4]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(7) {
// CHECK:             aie.packet_source<%[[TILE_6_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_3_5]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(8) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_4_2]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(9) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_4_3]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(10) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_4_4]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(11) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_4_5]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(12) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_5_2]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(13) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_5_3]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(14) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_5_4]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(15) {
// CHECK:             aie.packet_source<%[[TILE_7_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_5_5]], DMA : 1>
// CHECK:           }
// CHECK:         }
module @test_pktflow_weight_pusher {
  aie.device(xcvc1902) {
    %tile22 = aie.tile(2, 2) // 5'b0_0000
    %tile32 = aie.tile(3, 2) // 5'b0_0100
    %tile42 = aie.tile(4, 2) // 5'b0_1000
    %tile52 = aie.tile(5, 2) // 5'b0_1100

    %tile23 = aie.tile(2, 3) // 5'b0_0001
    %tile33 = aie.tile(3, 3) // 5'b0_0101
    %tile43 = aie.tile(4, 3) // 5'b0_1001
    %tile53 = aie.tile(5, 3) // 5'b0_1101

    %tile24 = aie.tile(2, 4) // 5'b0_0010
    %tile34 = aie.tile(3, 4) // 5'b0_0110
    %tile44 = aie.tile(4, 4) // 5'b0_1010
    %tile54 = aie.tile(5, 4) // 5'b0_1110

    %tile25 = aie.tile(2, 5) // 5'b0_0011
    %tile35 = aie.tile(3, 5) // 5'b0_0111
    %tile45 = aie.tile(4, 5) // 5'b0_1011
    %tile55 = aie.tile(5, 5) // 5'b0_1111

    // Herd "weight"
    %tile65 = aie.tile(6, 5)
    %tile75 = aie.tile(7, 5)


    // Tile (6, 5) streams data to the first two columns of herd "compute"
    // Tile (7, 5) streams data to the next two columns of herd "compute"
    //
    //  (2, 5)--(3, 5)--(4, 5)--(5, 5) < --(6, 5) <-- (7, 5)
    //    |       |       |       |
    //  (2, 4)--(3, 4)--(4, 4)--(5, 4)
    //    |       |       |       |
    //  (2, 3)--(3, 3)--(4, 3)--(5, 3)
    //    |       |       |       |
    //  (2, 2)--(3, 2)--(4, 2)--(5, 2)
    //

    // weight[0]: 0 - 7
    aie.packet_flow(0x0) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile22, DMA : 1>
    }

    aie.packet_flow(0x1) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile23, DMA : 1>
    }

    aie.packet_flow(0x2) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile24, DMA : 1>
    }

    aie.packet_flow(0x3) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile25, DMA : 1>
    }

    aie.packet_flow(0x4) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile32, DMA : 1>
    }

    aie.packet_flow(0x5) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile33, DMA : 1>
    }

    aie.packet_flow(0x6) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile34, DMA : 1>
    }

    aie.packet_flow(0x7) {
      aie.packet_source<%tile65, DMA : 0>
      aie.packet_dest<%tile35, DMA : 1>
    }

    // weight[1]: 8 - 15
    aie.packet_flow(0x8) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile42, DMA : 1>
    }

    aie.packet_flow(0x9) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile43, DMA : 1>
    }

    aie.packet_flow(0xa) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile44, DMA : 1>
    }

    aie.packet_flow(0xb) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile45, DMA : 1>
    }

    aie.packet_flow(0xc) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile52, DMA : 1>
    }

    aie.packet_flow(0xd) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile53, DMA : 1>
    }

    aie.packet_flow(0xe) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile54, DMA : 1>
    }

    aie.packet_flow(0xf) {
      aie.packet_source<%tile75, DMA : 0>
      aie.packet_dest<%tile55, DMA : 1>
    }

  }
}
