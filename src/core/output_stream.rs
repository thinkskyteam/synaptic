use crate::core::load_model::deserialize_weight_map;
use candle_core::Result;
use serde::Deserialize;
use std::collections::HashSet;

/// A stream for processing and decoding tokens using a tokenizer.
///
/// The `TokenOutputStream` struct manages a sequence of tokens and provides
/// methods to decode them into human-readable strings. It keeps track of
/// the current position in the token stream and allows for incremental
/// decoding of tokens as they are received.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    /// Creates a new `TokenOutputStream` with the specified tokenizer.
    ///
    /// # Parameters
    ///
    /// - `tokenizer`: An instance of `tokenizers::Tokenizer` used for
    ///   encoding and decoding tokens.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `TokenOutputStream`.
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    /// Consumes the `TokenOutputStream` and returns the underlying tokenizer.
    ///
    /// # Returns
    ///
    /// Returns the `tokenizers::Tokenizer` instance contained within the
    /// `TokenOutputStream`.
    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    /// Decodes a slice of tokens into a string.
    ///
    /// # Parameters
    ///
    /// - `tokens`: A slice of token IDs to decode.
    ///
    /// # Returns
    ///
    /// Returns a `Result<String>` containing the decoded string if successful,
    /// or an error if decoding fails.
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    /// Processes the next token and returns any new text generated.
    ///
    /// This method updates the internal state with the provided token and
    /// checks if it generates new text compared to the previous state.
    ///
    /// # Parameters
    ///
    /// - `token`: The token ID to process.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Option<String>>`, where `Some(String)` contains
    /// the newly generated text if applicable, or `None` if no new text
    /// was generated.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decodes any remaining tokens and returns new text generated.
    ///
    /// This method checks if there is any new text generated from the
    /// remaining tokens since the last decoding.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Option<String>>`, where `Some(String)` contains
    /// the newly generated text if applicable, or `None` if no new text
    /// was generated.
    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decodes all tokens in the stream into a single string.
    ///
    /// # Returns
    ///
    /// Returns a `Result<String>` containing the decoded string of all
    /// tokens in the stream.
    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    /// Retrieves the token ID for a given string representation.
    ///
    /// # Parameters
    ///
    /// - `token_s`: A string slice representing the token to look up.
    ///
    /// # Returns
    ///
    /// Returns an `Option<u32>` containing the token ID if found, or `None`
    /// if the token does not exist in the vocabulary.
    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    /// Returns a reference to the underlying tokenizer.
    ///
    /// # Returns
    ///
    /// Returns a reference to the `tokenizers::Tokenizer` instance.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Clears the token stream and resets the indices.
    ///
    /// This method removes all tokens from the stream and resets the
    /// previous and current index counters to zero.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

/// Represents a collection of weight maps.
///
/// This struct is used to deserialize a JSON object containing weight maps.
/// It contains a single field, `weight_map`, which is a set of strings
/// representing the names or identifiers of the weight maps.
///
/// # Fields
///
/// - `weight_map`: A `HashSet<String>` that holds the unique identifiers
///   of the weight maps. This field is populated by deserializing a JSON
///   object using a custom deserialization function.
///
/// The `deserialize_weight_map` function is used to handle the deserialization
/// of the `weight_map` field, ensuring that it is correctly extracted from
/// the input JSON.
#[derive(Debug, Deserialize)]
pub(crate) struct WeightMaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    pub(crate) weight_map: HashSet<String>,
}
