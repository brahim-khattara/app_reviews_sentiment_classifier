// ignore_for_file: prefer_const_constructors, use_key_in_widget_constructors

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_localizations/flutter_localizations.dart';

void main() {
  runApp(SentimentApp());
}

class SentimentApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        scaffoldBackgroundColor: Colors.white,
      ),
      locale: Locale('ar', 'AE'),
      localizationsDelegates: [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate, // Added for better RTL support
      ],
      supportedLocales: [
        const Locale('ar', 'AE'),
      ],
      home: SentimentScreen(),
    );
  }
}

class SentimentScreen extends StatefulWidget {
  @override
  _SentimentScreenState createState() => _SentimentScreenState();
}

class _SentimentScreenState extends State<SentimentScreen> {
  final TextEditingController _controller = TextEditingController();
  String _sentiment = '';
  bool _isLoading = false;
  String _errorMessage = '';
  String _selectedVectorizer = 'tfidf';
  List<Map<String, String>> _neighborReviews = [];

  // Enhanced vectorizer options with Arabic labels
  final List<Map<String, String>> _vectorizerOptions = [
    {'value': 'tfidf', 'label': 'TF-IDF تحليل'},
    {'value': 'word2vec', 'label': 'Word2Vec تحليل'},
    {'value': 'fasttext', 'label': 'FastText تحليل'},
  ];

  Future<void> classifyPhrase(String phrase, String vectorizerType) async {
    setState(() {
      _isLoading = true;
      _errorMessage = '';
      _neighborReviews = [];
    });

    try {
      final url = Uri.parse('http://192.168.1.13:5000/classify');
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({
          'phrase': phrase,
          'vectorizer_type': vectorizerType,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _sentiment = data['sentiment'];
          if (data['neighbors'] != null) {
            final reviews = data['neighbors']['reviews'] as List;
            final sentiments = data['neighbors']['sentiments'] as List;
            _neighborReviews = List.generate(
              reviews.length,
              (i) => {
                'review': reviews[i].toString(),
                'sentiment': sentiments[i].toString(),
              },
            );
          }
        });
      } else {
        setState(() {
          _errorMessage = 'فشل في تصنيف العبارة. الرجاء المحاولة مرة أخرى.';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'خطأ في الاتصال: ${e.toString()}';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        title: const Text(
          'تحليل المراجعة',
          style: TextStyle(
            color: Colors.black87,
            fontSize: 20,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              const Text(
                'شارك تجربتك',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'ما رأيك في التطبيق؟',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey[600],
                ),
              ),
              const SizedBox(height: 20),
              // Vectorizer selection with improved styling
              Container(
                padding: EdgeInsets.symmetric(horizontal: 16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.grey[200]!),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    isExpanded: true,
                    value: _selectedVectorizer,
                    onChanged: (String? newValue) {
                      setState(() {
                        _selectedVectorizer = newValue!;
                      });
                    },
                    items: _vectorizerOptions.map<DropdownMenuItem<String>>((Map<String, String> option) {
                      return DropdownMenuItem<String>(
                        value: option['value'],
                        child: Text(
                          option['label']!,
                          style: TextStyle(fontSize: 16),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              TextField(
                controller: _controller,
                maxLines: 5,
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Colors.grey[50],
                  hintText: 'اكتب رأيك هنا...',
                  hintStyle: TextStyle(color: Colors.grey[400]),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(16),
                    borderSide: BorderSide(color: Colors.grey[200]!),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(16),
                    borderSide: BorderSide(color: Colors.grey[200]!),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(16),
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _isLoading
                    ? null
                    : () async {
                        FocusScope.of(context).unfocus();
                        if (_controller.text.isNotEmpty) {
                          await classifyPhrase(_controller.text, _selectedVectorizer);
                        }
                      },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  minimumSize: const Size(double.infinity, 54),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  elevation: 0,
                ),
                child: _isLoading
                    ? SizedBox(
                        height: 24,
                        width: 24,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                    : const Text(
                        'تحليل المراجعة',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
              ),
              if (_sentiment.isNotEmpty) ...[
                const SizedBox(height: 40),
                Container(
                  padding: EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.1),
                        spreadRadius: 1,
                        blurRadius: 10,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'نتيجة التحليل',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w600,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 24),
                      Row(
                        children: [
                          Icon(
                            _sentiment == 'Positive' 
                                ? Icons.sentiment_satisfied_alt
                                : Icons.sentiment_dissatisfied,
                            color: _sentiment == 'Positive'
                                ? Colors.green[700]
                                : Colors.red[700],
                            size: 32,
                          ),
                          SizedBox(width: 12),
                          Text(
                            _sentiment == 'Positive'
                                ? 'مراجعة إيجابية'
                                : 'مراجعة سلبية',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                              color: _sentiment == 'Positive'
                                  ? Colors.green[700]
                                  : Colors.red[700],
                            ),
                          ),
                        ],
                      ),
                      if (_neighborReviews.isNotEmpty) ...[
                        const SizedBox(height: 24),
                        Text(
                          'المراجعات المشابهة',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w600,
                            color: Colors.black87,
                          ),
                        ),
                        const SizedBox(height: 12),
                        ...List.generate(_neighborReviews.length, (index) {
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 8.0),
                            child: Card(
                              child: ListTile(
                                title: Text(_neighborReviews[index]['review']!),
                                trailing: Icon(
                                  _neighborReviews[index]['sentiment'] == 'Positive'
                                      ? Icons.thumb_up
                                      : Icons.thumb_down,
                                  color: _neighborReviews[index]['sentiment'] == 'Positive'
                                      ? Colors.green[700]
                                      : Colors.red[700],
                                ),
                              ),
                            ),
                          );
                        }),
                      ],
                    ],
                  ),
                ),
              ],
              if (_errorMessage.isNotEmpty) ...[
                const SizedBox(height: 20),
                Container(
                  padding: EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.red[50],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.error_outline, color: Colors.red),
                      SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          _errorMessage,
                          style: TextStyle(
                            color: Colors.red[700],
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}