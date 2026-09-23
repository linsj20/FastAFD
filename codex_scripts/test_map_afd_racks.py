import unittest
from collections import Counter

from map_afd_racks import map_racks


def inventory(racks, trays):
    return [dict(host=f"r{rack:02}-t{tray:02}", address=f"10.0.{rack}.{tray}",
                 rack=f"r{rack:02}", gpus=4)
            for rack in range(racks) for tray in range(trays)]


class RackMappingTest(unittest.TestCase):
    def check_mapping(self, racks, trays):
        nodes = inventory(racks, trays)
        ffn_racks = [f"r{rack:02}" for rack in range(16)]
        result = map_racks(list(reversed(nodes)), ffn_racks)
        self.assertEqual(result, map_racks(nodes, ffn_racks))
        self.assertEqual((result["attention_gpus"], result["ffn_gpus"]), (560, 16))
        self.assertEqual(result["fanin_per_ffn"], 35)
        ffn = result["ffn"]
        self.assertEqual([slot["rack"] for slot in ffn], ffn_racks)
        slots = result["attention"] + ffn
        self.assertEqual(len({(s["host"], s["slot"]) for s in slots}), 576)
        for slot in result["attention"]:
            destination = slot["dp_rank"] * 16 // 560
            self.assertEqual(destination, slot["ffn_dp_rank"])
            if slot["rack"] in ffn_racks:
                self.assertEqual(slot["rack"], ffn[destination]["rack"])
        for rack in range(16, racks):
            distribution = Counter(s["ffn_dp_rank"] for s in result["attention"]
                                   if s["rack"] == f"r{rack:02}")
            self.assertEqual(distribution, Counter({rank: 2 for rank in range(16)}))
        self.assertEqual(len(result["environment"]["MINISGL_AFD_ATTN_NODES"].split(",")), 560)

    def test_18_racks_32_gpus(self):
        self.check_mapping(18, 8)

    def test_16_racks_36_gpus(self):
        self.check_mapping(16, 9)

    def test_16_racks_32_gpus(self):
        result = map_racks(inventory(16, 8), [f"r{rack:02}" for rack in range(16)])
        self.assertEqual((result["attention_gpus"], result["ffn_gpus"],
                          result["fanin_per_ffn"]), (496, 16, 31))
        for slot in result["attention"]:
            self.assertEqual(slot["rack"], result["ffn"][slot["ffn_dp_rank"]]["rack"])

    def test_reject_invalid_inventory(self):
        nodes = inventory(18, 8)
        ffn_racks = [f"r{rack:02}" for rack in range(16)]
        for invalid, racks in [(nodes + [nodes[0]], ffn_racks),
                               (nodes[1:], ffn_racks),
                               (nodes[:-1], ffn_racks),
                               (nodes, ffn_racks + ["r00"]),
                               (nodes, ["absent"])]:
            with self.subTest(racks=racks, nodes=len(invalid)):
                with self.assertRaises(ValueError):
                    map_racks(invalid, racks)

    def test_eight_racks_with_partially_used_last_tray(self):
        nodes = inventory(8, 5)
        for node in nodes:
            if node["host"].endswith("t04"):
                node["gpus"] = 2
        result = map_racks(nodes, [f"r{rack:02}" for rack in range(8)])
        self.assertEqual((result["attention_gpus"], result["ffn_gpus"],
                          result["fanin_per_ffn"]), (136, 8, 17))
        self.assertTrue(all(s["rack"] == result["ffn"][s["ffn_dp_rank"]]["rack"]
                            for s in result["attention"]))


if __name__ == "__main__":
    unittest.main()
