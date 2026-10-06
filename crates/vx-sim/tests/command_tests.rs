//! Comprehensive unit and integration tests for vx-sim Brigadier command subsystem.

use glam::{Vec2, Vec3};
use vx_sim::command::{
    CommandContext, CommandDispatcher, CommandSyntaxError, CoordinateArg, EntitySelector,
    SelectorType, StringReader, Vec3Arg, register_builtins,
};

#[test]
fn test_string_reader_primitives() {
    let mut reader =
        StringReader::new("  123 -45.5 true \"hello world\" 'quoted \\' test' unquoted_token");
    reader.skip_whitespace();

    assert_eq!(reader.read_i32().unwrap(), 123);
    reader.skip_whitespace();

    assert!((reader.read_f32().unwrap() - (-45.5)).abs() < 1e-5);
    reader.skip_whitespace();

    assert!(reader.read_bool().unwrap());
    reader.skip_whitespace();

    assert_eq!(reader.read_string().unwrap(), "hello world");
    reader.skip_whitespace();

    assert_eq!(reader.read_string().unwrap(), "quoted ' test");
    reader.skip_whitespace();

    assert_eq!(reader.read_unquoted_string(), "unquoted_token");
}

#[test]
fn test_string_reader_error_handling() {
    let mut reader = StringReader::new("\"unterminated");
    assert!(matches!(
        reader.read_quoted_string(),
        Err(CommandSyntaxError::UnterminatedQuote { .. })
    ));

    let mut reader = StringReader::new("\"invalid \\x escape\"");
    assert!(matches!(
        reader.read_quoted_string(),
        Err(CommandSyntaxError::InvalidEscape { escape: 'x', .. })
    ));

    let mut reader = StringReader::new("not_an_int");
    assert!(matches!(
        reader.read_i32(),
        Err(CommandSyntaxError::InvalidInteger { .. })
    ));

    let mut reader = StringReader::new("not_a_bool");
    assert!(matches!(
        reader.read_bool(),
        Err(CommandSyntaxError::InvalidBool { .. })
    ));
}

#[test]
fn test_coordinate_parsing_and_resolution() {
    // 1. Relative coordinates
    let mut reader = StringReader::new("~ ~5 ~-10.5");
    let vec = Vec3Arg::parse(&mut reader).expect("valid relative coords");
    assert_eq!(vec.x, CoordinateArg::Relative(0.0));
    assert_eq!(vec.y, CoordinateArg::Relative(5.0));
    assert_eq!(vec.z, CoordinateArg::Relative(-10.5));

    let origin = Vec3::new(100.0, 64.0, -200.0);
    let resolved = vec.resolve(origin, Vec2::ZERO);
    assert_eq!(resolved, Vec3::new(100.0, 69.0, -210.5));

    // 2. Absolute coordinates
    let mut reader = StringReader::new("12.5 70 -50");
    let vec = Vec3Arg::parse(&mut reader).expect("valid absolute coords");
    assert_eq!(vec.x, CoordinateArg::Absolute(12.5));
    assert_eq!(vec.y, CoordinateArg::Absolute(70.0));
    assert_eq!(vec.z, CoordinateArg::Absolute(-50.0));
    assert_eq!(
        vec.resolve(origin, Vec2::ZERO),
        Vec3::new(12.5, 70.0, -50.0)
    );

    // 3. Local coordinates
    let mut reader = StringReader::new("^ ^2 ^10");
    let vec = Vec3Arg::parse(&mut reader).expect("valid local coords");
    assert_eq!(vec.x, CoordinateArg::Local(0.0));
    assert_eq!(vec.y, CoordinateArg::Local(2.0));
    assert_eq!(vec.z, CoordinateArg::Local(10.0));

    // 4. Mixing local and relative coordinates must fail
    let mut reader = StringReader::new("^ ~5 10");
    assert!(Vec3Arg::parse(&mut reader).is_err());
}

#[test]
fn test_selector_parsing() {
    // 1. Player name
    let mut reader = StringReader::new("Alex");
    let sel = EntitySelector::parse(&mut reader).unwrap();
    assert_eq!(sel.selector_type, SelectorType::Named("Alex".to_string()));

    // 2. Simple target selector
    let mut reader = StringReader::new("@p");
    let sel = EntitySelector::parse(&mut reader).unwrap();
    assert_eq!(sel.selector_type, SelectorType::NearestPlayer);
    assert_eq!(sel.filters.limit, Some(1));

    // 3. Filtered target selector
    let mut reader = StringReader::new("@e[type=zombie,distance=..20,limit=5,sort=nearest]");
    let sel = EntitySelector::parse(&mut reader).unwrap();
    assert_eq!(sel.selector_type, SelectorType::AllEntities);
    assert_eq!(sel.filters.entity_type.as_deref(), Some("zombie"));
    assert!(!sel.filters.type_negated);
    assert_eq!(sel.filters.limit, Some(5));
    assert_eq!(sel.filters.sort.as_deref(), Some("nearest"));
    let dist = sel.filters.distance.unwrap();
    assert_eq!(dist.min, None);
    assert_eq!(dist.max, Some(20.0));
    assert!(dist.matches(15.0));
    assert!(!dist.matches(25.0));

    // 4. Negated type selector
    let mut reader = StringReader::new("@e[type=!cow]");
    let sel = EntitySelector::parse(&mut reader).unwrap();
    assert_eq!(sel.filters.entity_type.as_deref(), Some("cow"));
    assert!(sel.filters.type_negated);
}

#[test]
fn test_command_execution() {
    let mut dispatcher = CommandDispatcher::new();
    register_builtins(&mut dispatcher);

    let mut ctx = CommandContext::console();
    ctx.executor_name = "Player1".to_string();
    ctx.executor_pos = Vec3::new(10.0, 64.0, 20.0);

    // /help
    let out = dispatcher.execute("/help", &mut ctx);
    assert!(out.success);
    assert!(out.message.contains("Available commands"));

    // /time set night
    let out = dispatcher.execute("/time set night", &mut ctx);
    assert!(out.success);
    assert_eq!(out.message, "Set the time to 13000");

    // /time set 8500
    let out = dispatcher.execute("/time set 8500", &mut ctx);
    assert!(out.success);
    assert_eq!(out.message, "Set the time to 8500");

    // /weather rain
    let out = dispatcher.execute("/weather rain", &mut ctx);
    assert!(out.success);
    assert_eq!(out.message, "Set weather to rain");

    // /tp ~10 ~ ~
    let out = dispatcher.execute("/tp ~10 ~ ~", &mut ctx);
    assert!(out.success);
    assert!(
        out.message
            .contains("Teleported Player1 to 20.00, 64.00, 20.00")
    );

    // /give Player1 voxel:diamond 64
    let out = dispatcher.execute("/give Player1 voxel:diamond 64", &mut ctx);
    assert!(out.success);
    assert_eq!(out.message, "Gave 64 [voxel:diamond]");

    // /say Hello server!
    let out = dispatcher.execute("/say Hello server!", &mut ctx);
    assert!(out.success);
    assert_eq!(out.message, "[Player1] Hello server!");
}

#[test]
fn test_tab_completion_suggestions() {
    let mut dispatcher = CommandDispatcher::new();
    register_builtins(&mut dispatcher);

    // 1. Partial command root: "/wea"
    let sug = dispatcher.suggest("/wea", 4);
    assert_eq!(sug.start, 1);
    assert_eq!(sug.length, 3);
    assert_eq!(sug.candidates.len(), 1);
    assert_eq!(sug.candidates[0].value, "weather");

    // 2. After "/weather "
    let sug = dispatcher.suggest("/weather ", 9);
    assert_eq!(sug.start, 9);
    assert_eq!(sug.length, 0);
    let values: Vec<&str> = sug.candidates.iter().map(|c| c.value.as_str()).collect();
    assert!(values.contains(&"clear"));
    assert!(values.contains(&"rain"));
    assert!(values.contains(&"thunder"));

    // 3. After "/time "
    let sug = dispatcher.suggest("/time ", 6);
    let values: Vec<&str> = sug.candidates.iter().map(|c| c.value.as_str()).collect();
    assert!(values.contains(&"set"));
    assert!(values.contains(&"query"));

    // 4. After "/time set "
    let sug = dispatcher.suggest("/time set ", 10);
    let values: Vec<&str> = sug.candidates.iter().map(|c| c.value.as_str()).collect();
    assert!(values.contains(&"day"));
    assert!(values.contains(&"night"));
    assert!(values.contains(&"noon"));
    assert!(values.contains(&"midnight"));
}
