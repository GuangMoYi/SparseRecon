class CardTracker:
    def __init__(self, my_cards):
        """
        初始化记牌器，计算另外两个玩家的初始牌。
        :param my_cards: 一个列表，表示自己手中的牌，例如 ['A', '10', '10', 'W']。
        """
        # 初始化整个牌堆
        self.cards = {
            'W': 2, '2': 4, 'A': 4, 'K': 4, 'Q': 4, 'J': 4, '10': 4,
            '9': 4, '8': 4, '7': 4, '6': 4, '5': 4, '4': 4, '3': 4
        }

        # 移除自己手中的牌
        for card in my_cards:
            if card == '0':  # 将 '0' 转换为 '10'
                card = '10'
            if card in self.cards and self.cards[card] > 0:
                self.cards[card] -= 1

    def play_card(self, played_cards):
        """
        记录两个玩家的出牌并更新剩余牌的状态。
        :param played_cards: 一个字符串，包含被打出的牌，例如 "A10W"。
        """
        print(f"对方出牌: {list(played_cards)}")
        for card in played_cards:
            if card == '0':  # 将 '0' 转换为 '10'
                card = '10'
            if card in self.cards and self.cards[card] > 0:
                self.cards[card] -= 1
            else:
                print(f"警告: 卡片 {card} 无法再被出牌或不存在！")

    def display_status(self):
        """
        打印当前其他两个玩家的剩余牌信息。
        """
        print("当前其他两个玩家剩余牌情况:")
        card_names = list(self.cards.keys())
        remaining_counts = [str(self.cards[card]) for card in card_names]

        # 调整 '10' 后的空格对齐
        display_names = [name if name != '10' else '10' for name in card_names]
        adjusted_names = " ".join(f"{name: >2}" for name in display_names)
        adjusted_counts = " ".join(f"{count: >2}" for count in remaining_counts)

        print(adjusted_names)
        print(adjusted_counts)

if __name__ == "__main__":
    # 用户输入自己的初始手牌
    my_cards = list(input("请输入你的初始手牌(直接输入，例如 'A03W33J'): "))

    # 初始化记牌器
    tracker = CardTracker(my_cards)

    # 初始状态
    print("\n初始状态:")
    tracker.display_status()

    while True:
        # 输入对方玩家出的牌
        played_cards = input("请输入对方出的牌(直接输入，例如 'A0W'，输入 '退出' 结束程序): ")
        if played_cards == "退出":
            break

        # 更新牌局状态
        tracker.play_card(played_cards)

        # 显示更新后的状态
        print("\n更新后的状态:")
        tracker.display_status()

    # 程序结束时显示最终剩余牌
    print("\n游戏结束，最终剩余牌:")
    tracker.display_status()
