import unittest
from nlp_module.risk import FLAGS, risk_flags

CASES = {
    "guarantee": (["MWG bảo lãnh vô điều kiện cho Bách Hóa Xanh vay 2.000 tỷ", "Công ty bảo lãnh khoản vay của công ty con",
                   "HĐQT thông qua việc bảo lãnh cho công ty liên kết",
                   "Khoản vay của công ty con được công ty mẹ bảo lãnh"],
                  ["Ngân hàng nhận bảo lãnh phát hành cổ phiếu", "Ngân hàng cung cấp thư bảo lãnh thực hiện hợp đồng",
                   "được VCB bảo lãnh thanh toán", "đại lý bảo lãnh phát hành trái phiếu", "VCB đẩy mạnh dịch vụ bảo lãnh dự thầu"]),
    "seizure": (["Cơ quan thi hành án kê biên tài sản của công ty", "Ngân hàng phong tỏa tài khoản của doanh nghiệp",
                 "Tòa án ra quyết định kê biên cổ phiếu"],
                ["Công ty mua tài sản cố định mới"]),
    "prosecution": (["Chủ tịch bị khởi tố vì thao túng thị trường", "Cơ quan điều tra bắt tạm giam cựu tổng giám đốc",
                     "Công an khởi tố vụ án tại công ty"],
                    ["Công ty công bố kết quả kinh doanh quý 3",
                     "Ông bị bắt buộc phải nghỉ"]),
    "sanction": (["Công ty bị xử phạt hành chính vì công bố thông tin chậm", "UBCK phạt 200 triệu đồng đối với doanh nghiệp",
                  "Cổ phiếu bị đình chỉ giao dịch"],
                 ["Doanh nghiệp được vinh danh là nơi làm việc tốt nhất"]),
    "bond_default": (["Công ty chậm thanh toán trái phiếu đến hạn", "Khoản nợ trái phiếu quá hạn tăng mạnh",
                      "Doanh nghiệp không thanh toán được lãi trái phiếu"],
                     ["Công ty phát hành thành công 1.000 tỷ đồng trái phiếu"]),
}


class RiskTest(unittest.TestCase):
    def test_guarantee_flag(self):
        self.assertEqual(risk_flags("MWG bảo lãnh vô điều kiện cho Bách Hóa Xanh vay 2.000 tỷ"), ["guarantee"])

    def test_no_flag(self):
        self.assertEqual(risk_flags("MWG khai trương 50 cửa hàng"), [])

    def test_every_pattern_exercised(self):
        self.assertEqual(list(FLAGS), list(CASES))
        for key, (pos, neg) in CASES.items():
            for t in pos:
                self.assertIn(key, risk_flags(t), t)
                self.assertIn(key, risk_flags(t.upper()), t)   # case-insensitive
            for t in neg:
                self.assertNotIn(key, risk_flags(t), t)

    def test_order_follows_flags(self):
        self.assertEqual(risk_flags("bị khởi tố, kê biên tài sản"), ["seizure", "prosecution"])


if __name__ == "__main__":
    unittest.main()
