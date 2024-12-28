import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:swear_detect/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(SentimentApp());

    // Verify that the app starts with no sentiment displayed.
    expect(find.text('Positive Review'), findsNothing);
    expect(find.text('Negative Review'), findsNothing);

    // You could extend this test to interact with the text field and button
    // if needed for a more comprehensive test.
  });
}
