//! Integration tests for anvil combining, repairs, and container clicks.

use telos_sim::anvil::{
    ANVIL_SLOT_RESULT, AnvilInventory, anvil_container_click, combine_anvil_items,
};
use telos_sim::enchantment::EnchantmentKind;
use telos_sim::inventory::{
    ClickButton, ClickMode, ITEM_BOW, ITEM_ENCHANTED_BOOK, ITEM_IRON_HELMET, ITEM_IRON_PICKAXE,
    ITEM_IRON_SWORD, Inventory, ItemStack,
};

#[test]
fn test_anvil_combine_two_pickaxes() {
    let mut pick_a = ItemStack::new(ITEM_IRON_PICKAXE, 1);
    pick_a
        .enchantments
        .set_enchantment(EnchantmentKind::Efficiency, 3);
    pick_a
        .enchantments
        .set_enchantment(EnchantmentKind::Unbreaking, 2);

    let mut pick_b = ItemStack::new(ITEM_IRON_PICKAXE, 1);
    pick_b
        .enchantments
        .set_enchantment(EnchantmentKind::Efficiency, 3);
    pick_b
        .enchantments
        .set_enchantment(EnchantmentKind::Fortune, 2);

    let res = combine_anvil_items(pick_a, pick_b).expect("combination should succeed");
    assert_eq!(res.result.item, ITEM_IRON_PICKAXE);
    // Efficiency 3 + 3 = 4
    assert_eq!(
        res.result
            .enchantments
            .get_level(EnchantmentKind::Efficiency),
        4
    );
    assert_eq!(
        res.result
            .enchantments
            .get_level(EnchantmentKind::Unbreaking),
        2
    );
    assert_eq!(
        res.result.enchantments.get_level(EnchantmentKind::Fortune),
        2
    );
    assert!(res.level_cost >= 5);
}

#[test]
fn test_anvil_combine_bow_with_book() {
    let bow = ItemStack::new(ITEM_BOW, 1);

    let mut book = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
    book.enchantments.set_enchantment(EnchantmentKind::Power, 4);
    book.enchantments.set_enchantment(EnchantmentKind::Flame, 1);

    let res = combine_anvil_items(bow, book).expect("bow with book should succeed");
    assert_eq!(res.result.item, ITEM_BOW);
    assert_eq!(res.result.enchantments.get_level(EnchantmentKind::Power), 4);
    assert_eq!(res.result.enchantments.get_level(EnchantmentKind::Flame), 1);
}

#[test]
fn test_anvil_combine_books() {
    let mut book_a = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
    book_a
        .enchantments
        .set_enchantment(EnchantmentKind::Protection, 3);

    let mut book_b = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
    book_b
        .enchantments
        .set_enchantment(EnchantmentKind::Protection, 3);
    book_b
        .enchantments
        .set_enchantment(EnchantmentKind::Unbreaking, 2);

    let res = combine_anvil_items(book_a, book_b).expect("book combine should succeed");
    assert_eq!(res.result.item, ITEM_ENCHANTED_BOOK);
    assert_eq!(
        res.result
            .enchantments
            .get_level(EnchantmentKind::Protection),
        4
    );
    assert_eq!(
        res.result
            .enchantments
            .get_level(EnchantmentKind::Unbreaking),
        2
    );
}

#[test]
fn test_anvil_wrong_target_for_book() {
    // Sword cannot receive Protection
    let sword = ItemStack::new(ITEM_IRON_SWORD, 1);
    let mut book = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
    book.enchantments
        .set_enchantment(EnchantmentKind::Protection, 4);

    let res = combine_anvil_items(sword, book);
    assert!(res.is_none());
}

#[test]
fn test_anvil_container_full_flow() {
    let mut anvil = AnvilInventory::new();
    let mut player_inv = Inventory::default();

    // Place helmet in left
    let mut helm = ItemStack::new(ITEM_IRON_HELMET, 1);
    helm.enchantments
        .set_enchantment(EnchantmentKind::Protection, 2);
    player_inv.slots[9] = helm; // Storage slot 0 -> slot_idx 3 in container

    // QuickMove from slot 3 to anvil left
    anvil_container_click(
        &mut anvil,
        &mut player_inv,
        30,
        3,
        ClickButton::Left,
        ClickMode::QuickMove,
    )
    .unwrap();
    assert_eq!(anvil.left.item, ITEM_IRON_HELMET);
    assert!(player_inv.slots[9].is_empty());

    // Place book in right
    let mut book = ItemStack::new(ITEM_ENCHANTED_BOOK, 1);
    book.enchantments
        .set_enchantment(EnchantmentKind::Respiration, 3);
    player_inv.slots[10] = book; // Storage slot 1 -> slot_idx 4

    // QuickMove from slot 4 to anvil right
    anvil_container_click(
        &mut anvil,
        &mut player_inv,
        30,
        4,
        ClickButton::Left,
        ClickMode::QuickMove,
    )
    .unwrap();
    assert_eq!(anvil.right.item, ITEM_ENCHANTED_BOOK);
    assert!(player_inv.slots[10].is_empty());

    // Verify result is computed
    assert!(!anvil.result.is_empty());
    assert_eq!(
        anvil
            .result
            .enchantments
            .get_level(EnchantmentKind::Protection),
        2
    );
    assert_eq!(
        anvil
            .result
            .enchantments
            .get_level(EnchantmentKind::Respiration),
        3
    );

    let cost = anvil.level_cost;
    assert!(cost > 0);

    // Retrieve via quick move
    let levels_spent = anvil_container_click(
        &mut anvil,
        &mut player_inv,
        30,
        ANVIL_SLOT_RESULT,
        ClickButton::Left,
        ClickMode::QuickMove,
    )
    .unwrap();

    assert_eq!(levels_spent, Some(cost));
    assert!(anvil.left.is_empty());
    assert!(anvil.right.is_empty());
    assert!(anvil.result.is_empty());

    // Check player inventory received the enchanted helmet
    let received = player_inv.slots[9];
    assert_eq!(received.item, ITEM_IRON_HELMET);
    assert_eq!(
        received.enchantments.get_level(EnchantmentKind::Protection),
        2
    );
    assert_eq!(
        received
            .enchantments
            .get_level(EnchantmentKind::Respiration),
        3
    );
}
